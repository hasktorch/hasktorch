{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Generation where

import Control.Applicative (Alternative (many, (<|>)), Applicative (liftA2), empty)
import Control.Monad (MonadPlus, mfilter)
import Control.Monad.Identity (runIdentityT)
import Control.Monad.State (modify, MonadState (..), MonadTrans, StateT (..), evalStateT, lift)
import Control.Monad.Trans.Free (FreeF (Free, Pure), FreeT (FreeT), iterTM, runFreeT)
import Data.Foldable (asum)
import Data.Functor (($>))
import Data.Kind (Type)
import Data.List (nub, sortOn, uncons)
import qualified Data.Map as Map ((!))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (logSoftmax)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, T5GenerationInput (..), T5Input (..), T5Output (..), mkT5Input, t5EosTokenId)
import Torch.GraduallyTyped.NN.Transformer.T5.Small (T5Small)
import Torch.GraduallyTyped.NN.Transformer.T5.Vocab (t5Vocab)
import Torch.GraduallyTyped.Random (Generator, mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (expand)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison (Order (..), Sorted (..), sort)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), shape)
import qualified Torch.Tensor
import Prelude hiding (Word, words)
import System.IO.Unsafe (unsafePerformIO)

data IsFinished = Finished | Unfinished

data Beams a b where
  Beams ::
    forall a b.
    { finished :: [Hypothesis 'Finished a b],
      unfinished :: [Hypothesis 'Unfinished a b]
    } ->
    Beams a b
  deriving (Show)

data Hypothesis (isFinished :: IsFinished) a b where
  InitialHypothesis ::
    forall a b.
    Hypothesis 'Unfinished a b
  UnfinishedHypothesis ::
    forall a b.
    { currentToken :: a,
      currentScore :: Float,
      previousHypothesis :: Hypothesis 'Unfinished a b
    } ->
    Hypothesis 'Unfinished a b
  FinishedHypothesis ::
    forall a b.
    { finalToken :: a,
      finalScore :: Float,
      penultimateHypothesis :: Hypothesis 'Unfinished a b,
      finalValue :: b
    } ->
    Hypothesis 'Finished a b

deriving instance (Eq a, Eq b) => Eq (Hypothesis 'Unfinished a b)

deriving instance (Eq a, Eq b) => Eq (Hypothesis 'Finished a b)

deriving instance (Ord a, Ord b) => Ord (Hypothesis 'Unfinished a b)

deriving instance (Ord a, Ord b) => Ord (Hypothesis 'Finished a b)

deriving instance (Show a, Show b) => Show (Hypothesis 'Unfinished a b)

deriving instance (Show a, Show b) => Show (Hypothesis 'Finished a b)

getTokens :: forall a b. Hypothesis 'Unfinished a b -> [a]
getTokens InitialHypothesis = []
getTokens (UnfinishedHypothesis token _ previousHypothesis) = token : getTokens previousHypothesis

getScore :: forall a b. Hypothesis 'Unfinished a b -> Float
getScore InitialHypothesis = 0
getScore (UnfinishedHypothesis _ previousScore _) = previousScore

data SomeHypothesis a b = forall isFinished. SomeHypothesis {unSomeHypothesis :: Hypothesis isFinished a b}

beamSearch ::
  forall a b m.
  Monad m =>
  Int ->
  Int ->
  ([Hypothesis 'Unfinished a b] -> m [SomeHypothesis a b]) ->
  m [Beams a b]
beamSearch maxSteps beamSize cont = go maxSteps (Beams [] (replicate beamSize InitialHypothesis))
  where
    go !_ (Beams _ []) = pure []
    go n beams
      | n <= 0 = pure []
      | otherwise = do
        beams' <- beamSearchStep cont beams
        (beams' :) <$> go (n - 1) beams'

beamSearchStep ::
  forall a b m.
  Functor m =>
  ([Hypothesis 'Unfinished a b] -> m [SomeHypothesis a b]) ->
  Beams a b ->
  m (Beams a b)
beamSearchStep cont beam =
  let someHypotheses =
        take (length (unfinished beam))
          . reverse
          . sortOn @Float
            ( \case
                SomeHypothesis InitialHypothesis -> 0
                SomeHypothesis u@UnfinishedHypothesis {} -> currentScore u
                SomeHypothesis f@FinishedHypothesis {} -> finalScore f
            )
          <$> (cont . unfinished $ beam)
   in foldl
        ( \(Beams fs us) someHypothesis ->
            case someHypothesis of
              SomeHypothesis u@InitialHypothesis -> Beams fs (u : us)
              SomeHypothesis u@UnfinishedHypothesis {} -> Beams fs (u : us)
              SomeHypothesis f@FinishedHypothesis {} -> Beams (f : fs) us
        )
        (Beams (finished beam) [])
        <$> someHypotheses

foo ::
  forall model input decoderInput encoderOutput encoderOutputShape encoderOutput' inputPaddingMask decoderOutput generator.
  ( HasForward
      model
      (T5Input input decoderInput)
      generator
      (T5Output decoderOutput encoderOutput inputPaddingMask)
      generator,
    encoderOutput
      ~ Tensor
          'WithGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          T5DataType
          encoderOutputShape,
    'UncheckedShape ~ BroadcastShapesF encoderOutputShape 'UncheckedShape,
    KnownShape encoderOutputShape,
    HasForward
      model
      (T5GenerationInput decoderInput encoderOutput' inputPaddingMask)
      generator
      (T5Output decoderOutput encoderOutput' inputPaddingMask)
      generator,
    encoderOutput'
      ~ Tensor
          'WithGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          T5DataType
          'UncheckedShape,
    decoderInput
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[ 'Dim ( 'Name "*") 'UncheckedSize, 'Dim ( 'Name "*") 'UncheckedSize]),
    decoderOutput
      ~ Tensor
          'WithGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          T5DataType
          'UncheckedShape,
    generator ~ Generator ( 'Device 'CPU)
  ) =>
  Int ->
  Int ->
  model ->
  input ->
  generator ->
  IO [Beams Int [Int]]
foo maxSteps beamSize model input g =
  evalStateT (beamSearch maxSteps beamSize cont) (Nothing, g)
  where
    cont :: [Hypothesis 'Unfinished Int [Int]] -> StateT (Maybe (encoderOutput, inputPaddingMask), generator) IO [SomeHypothesis Int [Int]]
    cont previousHypotheses = do
      let previousHypotheses' = nub previousHypotheses
      decoderInput :: decoderInput <- do
        let tokens = reverse . getTokens <$> previousHypotheses'
            batchSize = fromIntegral $ length previousHypotheses'
            seqSize =
              let go :: Hypothesis 'Unfinished Int [Int] -> Int
                  go InitialHypothesis = 0
                  go (UnfinishedHypothesis _ _ previousHypothesis') = 1 + go previousHypothesis'
               in fromIntegral . maximum $ go <$> previousHypotheses'
        -- liftIO . print $ ((t5Vocab Map.!) <$>) <$> tokens
        mkT5Input @( 'Dim ( 'Name "*") 'UncheckedSize) @( 'Dim ( 'Name "*") 'UncheckedSize) batchSize seqSize tokens
      logProbs <- getLogProbs decoderInput
      pure $ zip previousHypotheses' logProbs >>= uncurry (\previousHypothesis -> zipWith (mkHypothesis previousHypothesis) [0, 1 ..] . last)
    getLogProbs :: decoderInput -> StateT (Maybe (encoderOutput, inputPaddingMask), generator) IO [[[Float]]]
    getLogProbs decoderInput = do
      (maybeStuff, g) <- get
      let (T5Output decoderOutput encoderOutput inputPaddingMask, g') = case maybeStuff of
            Nothing -> forward model (T5Input input decoderInput) g
            Just (encoderOutput, inputPaddingMask) ->
              let decoderInputBatchDim : _ = shape decoderInput
                  _encoderOutputBatchDim : encoderOutputDims = shape encoderOutput
                  encoderOutput' = expand @ 'UncheckedShape (decoderInputBatchDim : encoderOutputDims) encoderOutput
               in case forward model (T5GenerationInput decoderInput encoderOutput' inputPaddingMask) g of
                    (T5Output decoderOutput _ _, g') -> (T5Output decoderOutput encoderOutput inputPaddingMask, g')
      put (Just (encoderOutput, inputPaddingMask), g')
      case logSoftmax @( 'SelectDim ( 'ByIndex 2)) decoderOutput of
        UnsafeTensor t -> pure . Torch.Tensor.asValue . Torch.Tensor.Unsafe $ t
    mkHypothesis :: Hypothesis 'Unfinished Int [Int] -> Int -> Float -> SomeHypothesis Int [Int]
    mkHypothesis previousHypothesis token logProb
      | token == t5EosTokenId =
        let finalValue = reverse $ token : getTokens previousHypothesis
            finalScore = logProb + getScore previousHypothesis
         in SomeHypothesis (FinishedHypothesis token finalScore previousHypothesis finalValue)
      | otherwise =
        let score = logProb + getScore previousHypothesis
         in SomeHypothesis (UnfinishedHypothesis token score previousHypothesis)

bar = do
  input <- do
    let tokens = [[13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
    -- let tokens = [[13959, 1566, 12, 2968, 10, 148, 31, 60, 423, 13, 3, 7, 10536, 55, 1]]
    print $ ((t5Vocab Map.!) <$>) <$> tokens
    mkT5Input
      @( 'Dim ( 'Name "*") ( 'Size 1))
      @( 'Dim ( 'Name "*") ( 'Size 19))
      tokens
  model <-
    initialize
      @(T5Small 'WithLMHead ( 'Device 'CPU))
      "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
  g <- mkGenerator @( 'Device CPU) 0
  Beams finished _ <- last <$> foo 50 1 model input g
  print $ finalValue <$> finished
  let tmp = parseString @[] words . finalValue <$> finished
  print tmp

newtype WordPiece = WordPiece {unWordPiece :: String} deriving (Eq, Ord, Show)

newtype Word = Word {unWord :: String} deriving (Eq, Ord, Show)

word :: forall b. MonadPlus b => Parser b Int Word
word =
  let f (c, _)
        | c == '▁' = True
        | otherwise = False
      p (WordPiece ",") = True
      p (WordPiece ":") = True
      p (WordPiece ".") = True
      p (WordPiece ";") = True
      p (WordPiece "!") = True
      p (WordPiece "-") = True
      p (WordPiece "?") = True
      p (WordPiece "...") = True
      p (WordPiece s) = maybe False f (uncons s)
      p' (WordPiece "</s>") = False
      p' _ = True
      predicate p = mfilter p (WordPiece . (t5Vocab Map.!) <$> token)
      begin = predicate (\wp -> p' wp && p wp)
      append = predicate (\wp -> p' wp && not (p wp))
      strip ('▁' : s) = s
      strip s = s
      concat = strip . mconcat . (unWordPiece <$>)
   in Word . concat <$> liftA2 (:) begin (many append)

words :: forall b. MonadPlus b => Parser b Int [Word]
words =
  let predicate p' = mfilter p' (WordPiece . (t5Vocab Map.!) <$> token)
      is t = predicate (== t)
      end = is (WordPiece "</s>")
   in manyTill word end

type Parser (b :: Type -> Type) (i :: Type) (a :: Type) = FreeT ((->) i) b a

-- | recurse parser
-- @
-- recurse next parser = next parser (\parser' -> next parser' (\parser'' -> next parser'' (\parser''' -> next parser''' (...))))
-- @
recurse ::
  forall t b i a.
  (Parser b i a -> (Parser b i a -> t b a) -> t b a) ->
  Parser b i a ->
  t b a
recurse next parser =
  let cont = recurse next
   in next parser cont

next ::
  forall t b i a.
  (MonadTrans t, Monad b, Monad (t b)) =>
  Parser b i a ->
  (Parser b i a -> t b a) ->
  t b a
next parser cont = do
  val <- lift . runFreeT $ parser
  case val of
    Pure a -> pure a
    Free feed ->
      let i = undefined
       in cont (feed i)

next' ::
  forall t b i a.
  _ =>
  Parser (StateT [i] b) i a ->
  (Parser (StateT [i] b) i a -> t (StateT [i] b) a) ->
  t (StateT [i] b) a
next' parser cont = do
  val <- lift $ runFreeT parser
  case val of
    Pure a -> pure a
    Free feed ->
      let is :: StateT [i] b i = undefined
       in cont . feed =<< lift is

testNext' :: Parser (StateT [i] []) i a -> [(a, [i])]
testNext' = flip runStateT [] . runIdentityT . recurse next'

next'' ::
  forall s a model input decoderInput encoderOutput decoderOutput inputPaddingMask generator.
  ( s ~ (Maybe (encoderOutput, inputPaddingMask), generator),
    decoderInput
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape
              '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") 'UncheckedSize]
          ),
    decoderOutput
      ~ Tensor
          'WithGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          T5DataType
          'UncheckedShape,
    HasForward
      model
      (T5Input input decoderInput)
      generator
      (T5Output decoderOutput encoderOutput inputPaddingMask)
      generator,
    HasForward
      model
      (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
      generator
      (T5Output decoderOutput encoderOutput inputPaddingMask)
      generator
  ) =>
  model ->
  input ->
  Parser (StateT [Int] []) Int a ->
  (Parser (StateT [Int] []) Int a -> StateT s (StateT [Int] []) a) ->
  StateT s (StateT [Int] []) a
next'' model input parser cont = do
  -- currently every submitted token is tried for the current parser until moving on to the next parser.
  -- what we need to do instead is trying all parsers on each token before moving on to the next token.
  val <- lift $ runFreeT parser
  case val of
    Pure a -> pure a
    Free feed -> do
      i <- is
      let j = unsafePerformIO $ do
                let k = i
                putStrLn $ "current i: " <> show k
                pure k
      lift $ modify (j:)
      cont . feed $ j
  where
    is :: StateT s (StateT [Int] []) Int
    is = do
      -- tokens <- reverse <$> lift get
      tokens <- do
        ts <- reverse <$> lift get
        let ts' = unsafePerformIO $ do
              putStrLn $ "tokens: " <> show ((t5Vocab Map.!) <$> ts)
              pure ts
        pure ts'
      decoderInput :: decoderInput <- mkT5Input @( 'Dim ( 'Name "*") ( 'Size 1)) @( 'Dim ( 'Name "*") 'UncheckedSize) (fromIntegral $ length tokens) [tokens]
      decoderOutput <- do
        (mTensors, g) <- get
        let (T5Output decoderOutput encoderOutput inputPaddingMask, g') =
              case mTensors of
                Nothing -> forward model (T5Input input decoderInput) g
                Just (encoderOutput, inputPaddingMask) ->
                  forward model (T5GenerationInput decoderInput encoderOutput inputPaddingMask) g
        put (Just (encoderOutput, inputPaddingMask), g')
        pure decoderOutput
      case sort @( 'SelectDim ( 'ByIndex 2)) Descending
        . logSoftmax @( 'SelectDim ( 'ByIndex 2))
        $ decoderOutput of
        Sorted _ (UnsafeTensor indices) ->
          let indices' = -- take 3 . last . head . Torch.Tensor.asValue @[[[Int]]] . Torch.Tensor.Unsafe $ indices
                unsafePerformIO $ do
                  let tmp = take 15 . last . head . Torch.Tensor.asValue @[[[Int]]] . Torch.Tensor.Unsafe $ indices
                  putStrLn $ "top-2: " <> show (take 2 tmp)
                  -- putStrLn $ "bottom-5: " <> show (reverse $ take 5 $ reverse tmp)
                  pure tmp
          in  lift . lift $ indices'

testNext'' ::
  forall model input generator a encoderOutput inputPaddingMask.
  _ =>
  model ->
  input ->
  generator ->
  Parser (StateT [Int] []) Int a ->
  [((a, (Maybe (encoderOutput, inputPaddingMask), generator)), [Int])]
testNext'' model input g = flip runStateT [] . flip runStateT (Nothing, g) . recurse (next'' model input)

bar' = do
  input <- do
    let tokens = [[13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
    -- let tokens = [[13959, 1566, 12, 2968, 10, 148, 31, 60, 423, 13, 3, 7, 10536, 55, 1]]
    print $ ((t5Vocab Map.!) <$>) <$> tokens
    mkT5Input
      @( 'Dim ( 'Name "*") ( 'Size 1))
      @( 'Dim ( 'Name "*") ( 'Size 19))
      tokens
  model <-
    initialize
      @(T5Small 'WithLMHead ( 'Device 'CPU))
      "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
  g <- mkGenerator @( 'Device CPU) 0
  let outputs = testNext'' model input g words
  print (take 1 $ fst . fst <$> outputs)

parseStream :: forall s b i a. Monad b => (s -> b (i, s)) -> Parser b i a -> s -> b (a, s)
parseStream next = runStateT . iterTM (StateT next >>=)

parseString :: forall b i a. MonadPlus b => Parser b i a -> [i] -> b (a, [i])
parseString = parseStream (maybe empty pure . uncons)

token :: forall b i. Applicative b => Parser b i i
token = FreeT . pure . Free $ FreeT . pure . Pure

predicate :: forall b i. (MonadPlus b) => (i -> Bool) -> Parser b i i
predicate p = mfilter p token

is :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
is t = predicate (== t)

isNot :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
isNot t = predicate (/= t)

choice :: Alternative f => [f a] -> f a
choice = asum

option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan where scan = (end $> []) <|> liftA2 (:) p scan

skipMany :: Alternative f => f a -> f ()
skipMany p = scan where scan = (p *> scan) <|> pure ()

skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p
