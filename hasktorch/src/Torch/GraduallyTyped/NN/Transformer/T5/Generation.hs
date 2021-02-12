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
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Generation where

import Control.Applicative (Alternative (..), Applicative (..))
import Control.Monad (MonadPlus (..), guard, mfilter, replicateM)
import Control.Monad.Logic (observe)
import Control.Monad.State (MonadState (..), MonadTrans (..), StateT (..), evalStateT, gets, lift, modify)
import Control.Monad.Trans.Free (FreeF (..), FreeT (..), iterTM, runFreeT)
import Data.Foldable (asum)
import Data.Functor (($>))
import Data.Kind (Type)
import Data.List (nub, sortOn, uncons)
import qualified Data.Map as Map ((!))
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (logSoftmax)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Base (T5Base)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, T5GenerationInput (..), T5Input (..), T5Output (..), mkT5Input, t5EosTokenId)
import Torch.GraduallyTyped.NN.Transformer.T5.Large (T5Large)
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

runBeamSearch ::
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
runBeamSearch maxSteps beamSize model input g =
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

testBeamSearch = do
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
  Beams finished _ <- last <$> runBeamSearch 50 1 model input g
  print $ finalValue <$> finished
  let tmp = parseString @[] words . finalValue <$> finished
  print tmp

-- | A @Parser b i a@ is a parser that consumes a stream of @i@ tokens and as a
-- result yields a value of type @a@, while operating under the @b@
-- non-determinism monad.
--
-- For most purposes, the non-determinism monad @b@ should be a 'MonadPlus'.
-- Useful examples include @[]@ or @Logic@ if you want backtracking,
-- and 'Maybe' if you want no backtracking.
--
-- Use @'StateT' s []@ if you want to maintain a state @s@ that is
-- automatically reverted when backtracking via @[]@.
--
-- 'hoistFreeT' can be used to change the backtracking monad.
--
-- 'FreeT' provides instances for 'Functor', 'Applicative', 'Monad',
-- 'Alternative' and 'MonadPlus'.
type Parser
  (b :: Type -> Type)
  (i :: Type)
  (a :: Type) =
  FreeT ((->) i) b a

-- | Recurse over a parser.
--
-- Tears down the free monad transformer over the '(->) i' functor using iteration:
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
  ( i ~ Int,
    Show i,
    MonadTrans t,
    Monad (t (StateT [i] b)),
    Alternative (t (StateT [i] b)),
    Monad b,
    Foldable b
  ) =>
  t (StateT [i] b) i ->
  Parser (StateT [i] b) i a ->
  (Parser (StateT [i] b) i a -> t (StateT [i] b) a) ->
  t (StateT [i] b) a
next is parser cont = do
  lift (notNull parser) >>= guard
  i <- is
  lift $ modify (i :)
  val <- lift $ runFreeT parser
  case val of
    Pure a -> pure a
    Free feed -> unsafePerformIO $ do
      putStrLn $ "feed: " <> show (t5Vocab Map.! i)
      pure $ cont . feed $ i

notNull ::
  (Monad b, Foldable b) =>
  Parser (StateT [i] b) i a ->
  StateT [i] b Bool
notNull parser = gets $ (not . null) . evalStateT (runFreeT parser)

hasFree ::
  (Monad b, Foldable b) =>
  Parser (StateT [i] b) i a ->
  StateT [i] b Bool
hasFree parser = gets $ any (\case Free _ -> True; _ -> False) . evalStateT (runFreeT parser)

getIs ::
  forall model input generator b decoderInput encoderOutput decoderOutput inputPaddingMask s.
  ( Alternative b,
    MonadFail b,
    s ~ (Maybe (encoderOutput, inputPaddingMask), generator),
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
  StateT s (StateT [Int] b) Int
getIs model input = do
  -- tokens <- reverse <$> lift get
  tokens <- do
    ts <- reverse <$> lift get
    let ts' = unsafePerformIO $ do
          putStrLn $ "tokens: " <> show ((t5Vocab Map.!) <$> ts)
          pure ts
    pure ts'
  decoderInput :: decoderInput <-
    mkT5Input
      @( 'Dim ( 'Name "*") ( 'Size 1))
      @( 'Dim ( 'Name "*") 'UncheckedSize)
      (fromIntegral $ length tokens)
      [tokens]
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
      let indices' = last . head . Torch.Tensor.asValue @[[[Int]]] . Torch.Tensor.Unsafe $ indices
       in lift . lift . asum $ pure <$> indices'

runParser ::
  forall model input generator b a.
  _ =>
  model ->
  input ->
  generator ->
  Parser (StateT [Int] b) Int a ->
  b (a, [Int])
runParser model input g =
  flip runStateT []
    . flip evalStateT (Nothing, g)
    . recurse (next (getIs model input))

testParser = do
  input <- do
    -- let tokens = [[13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
    -- let tokens = [[13959, 1566, 12, 2968, 10, 148, 31, 60, 423, 13, 3, 7, 10536, 55, 1]]
    -- let tokens = [[13959, 1566, 12, 2968, 10, 3, 31, 7, 15, 3437, 3, 17, 4416, 4350, 6, 3476, 599, 1935, 61, 45, 4219, 38, 3, 17, 536, 1715, 14939, 38, 3, 17, 357, 30, 3, 17, 5411, 2427, 12925, 834, 23, 26, 3274, 3, 17, 4416, 2427, 12925, 834, 23, 26, 563, 57, 3, 17, 5411, 2427, 12925, 834, 23, 26, 31, 1]]
    let tokens = [[13959, 1566, 12, 2968, 10, 96, 3, 23143, 14196, 332, 4416, 4350, 6, 2847, 17161, 599, 1935, 61, 21680, 4219, 6157, 332, 536, 3, 15355, 3162, 14939, 6157, 332, 357, 9191, 332, 5411, 2427, 12925, 834, 23, 26, 3274, 332, 4416, 2427, 12925, 834, 23, 26, 350, 4630, 6880, 272, 476, 3, 17, 5411, 2427, 12925, 834, 23, 26, 96, 1]]
    print $ length <$> tokens
    print $ ((t5Vocab Map.!) <$>) <$> tokens
    mkT5Input
      @( 'Dim ( 'Name "*") ( 'Size 1))
      @( 'Dim ( 'Name "*") ( 'Size 61))
      tokens
  model <-
    initialize
      @(T5Small 'WithLMHead ( 'Device 'CPU))
      "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
  g <- mkGenerator @( 'Device CPU) 0
  let outputs = runParser model input g words
  pure . fst $ observe outputs

parseStream :: forall s b i a. Monad b => (s -> b (i, s)) -> Parser b i a -> s -> b (a, s)
parseStream next = runStateT . iterTM (StateT next >>=)

parseString :: forall b i a. MonadPlus b => Parser b i a -> [i] -> b (a, [i])
parseString = parseStream (maybe empty pure . uncons)

-- | @token@ is trivial parser that consumes a single token @i@ and yields it.
--
-- Other parsers can be derived from this one using methods of the
-- 'Functor', 'Applicative', 'Monad', 'Alternative', and 'MonadPlus' typeclasses
-- and the parser combinators in this module.
token :: forall b i. Applicative b => Parser b i i
token = FreeT . pure . Free $ FreeT . pure . Pure

-- | @satisfy p@ is a simple parser that consumes a single token @i@ and yields it
-- if and only if @p i@ evaluates to 'True'. Otherwise, the parser fails.
satisfy :: forall b i. MonadPlus b => (i -> Bool) -> Parser b i i
satisfy p = mfilter p token

-- | @is i@ is a simple parser that consumes a single token and yields it
-- if and only if it is equal to @i@. Otherwise, the parser fails.
is :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
is i = satisfy (== i)

-- | @isNot i@ is a simple parser that consumes a single token and yields it
-- if and only if it is not equal to @i@. If the token is equal to @i@,
-- the parser fails.
isNot :: forall b i. (MonadPlus b, Eq i) => i -> Parser b i i
isNot i = satisfy (/= i)

-- | @choice ps@ tries to apply the parsers in the list @ps@ in order,
-- until one of them succeeds. Returns the value of the succeeding
-- parser.
choice :: Alternative f => [f a] -> f a
choice = asum

-- | @option a p@ tries to apply parser @p@. If the parser @p@ fails,
-- it returns the value @a@, otherwise the value returned by the parser @p@.
option :: Alternative f => a -> f a -> f a
option a p = p <|> pure a

-- | @many1 p@ applies the parser @p@ /one/ or more times. Returns a
-- list of the returned values of @p@.
many1 :: Alternative f => f a -> f [a]
many1 p = liftA2 (:) p (many p)
{-# INLINE many1 #-}

-- | @manyTill p end@ applies the parser @p@ /zero/ or more times until
-- the parser @end@ succeeds, and returns the list of values returned by
-- @p@. The result of @end@ is discarded.
--
-- Note that this can be inefficient if the parsers @p@ and @end@ overlap,
-- as it can lead to a lot of backtracking.
manyTill :: Alternative f => f a -> f b -> f [a]
manyTill p end = scan where scan = (end $> []) <|> liftA2 (:) p scan

-- | @manyTill p end@ applies the parser @p@ /one/ or more times until
-- the parser @end@ succeeds, and returns the list of values returned by
-- @p@. The result of @end@ is discarded.
--
-- Note that this can be inefficient if the parsers @p@ and @end@ overlap,
-- as it can lead to a lot of backtracking.
many1Till :: Alternative f => f a -> f b -> f [a]
many1Till p end = liftA2 (:) p (manyTill p end)

-- | @repeat n p@ applies the parser @p@ @n@ times and returns
-- every parsing result. If parsing of @p@ succeeds less the @n@ times,
-- @repeat n p@ fails.
repeat :: Monad m => Int -> m a -> m [a]
repeat = replicateM

-- | @skipMany p@ skips /zero/ or more instances of the parser @p@.
-- The parsing results are discarded.
skipMany :: Alternative f => f a -> f ()
skipMany p = scan where scan = (p *> scan) <|> pure ()

-- | @skipMany1 p@ skips /one/ or more instances of the parser @p@.
-- The parsing results are discarded.
skipMany1 :: Alternative f => f a -> f ()
skipMany1 p = p *> skipMany p

-- | @sepBy p sep@ applies /zero/ or more occurrences of the parser @p@,
-- separated by @sep@. Returns a list of the values returned by @p@ and
-- discards the results of @sep@.
sepBy :: Alternative f => f a -> f sep -> f [a]
sepBy p sep = liftA2 (:) p ((sep *> sepBy1 p sep) <|> pure []) <|> pure []

-- | @sepBy1 p sep@ applies /one/ or more occurrences of the parser @p@,
-- separated by @sep@. Returns a list of the values returned by @p@ and
-- discards the results of @sep@.
sepBy1 :: Alternative f => f a -> f sep -> f [a]
sepBy1 p sep = scan where scan = liftA2 (:) p ((sep *> scan) <|> pure [])

-- | @maybeP p@ applies the parser @p@ optionally and returns the result
-- wrapped in @Maybe@.
maybeP :: Alternative f => f a -> f (Maybe a)
maybeP p = option Nothing (Just <$> p)

-- | @eitherP p p'@ combines the two alternatives @p@ and @p'@.
eitherP :: Alternative f => f a -> f b -> f (Either a b)
eitherP p p' = (Left <$> p) <|> (Right <$> p')

-- | @void p@ applies the parser @p@ and discards its result.
void :: Functor f => f a -> f ()
void p = p $> ()

-- | @between open close p@ applies the parsers @open@, @p@, and @close@
-- in that order. Only the result of @p@ is returned, the results of @open@
-- and @close@ are discarded.
--
-- This combinator is useful for parsing expressions wrapped in parentheses,
-- for example.
between :: Applicative f => f a1 -> f a2 -> f a -> f a
between open close p = open *> p <* close

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
      satisfy p = mfilter p (WordPiece . (t5Vocab Map.!) <$> token)
      begin = satisfy ((&&) <$> p' <*> p)
      append = satisfy ((&&) <$> p' <*> (not . p))
      strip ('▁' : s) = s
      strip s = s
      concat = strip . mconcat . (unWordPiece <$>)
   in Word . concat <$> ((:) <$> begin <*> many append)

words :: forall b. MonadPlus b => Parser b Int [Word]
words =
  let satisfy p' = mfilter p' (WordPiece . (t5Vocab Map.!) <$> token)
      is t = satisfy (== t)
      end = is (WordPiece "</s>")
      forceBeginning :: FreeT ((->) Int) b Word =
        let rewrap (WordPiece ('▁' : s)) = Word s
         in rewrap <$> is (WordPiece "\9601Wer")
   in -- (:) <$> forceBeginning <*>
      manyTill word end

data Sql = Sql
  { sqlSelect :: Select,
    sqlFrom :: From,
    sqlWhere :: Maybe Cond,
    sqlGroupBy :: [ColUnit],
    sqlOrderBy :: Maybe OrderBy,
    sqlHaving :: Maybe Cond,
    sqlLimit :: Maybe Int,
    sqlIntersect :: Maybe Sql,
    sqlExcept :: Maybe Sql,
    sqlUnion :: Maybe Sql
  }

data Select
  = Select [Agg]
  | SelectDistinct [Agg]

data From = From
  { fromTableUnits :: [TableUnit],
    fromCond :: Maybe Cond
  }

data Cond
  = And Cond Cond
  | Or Cond Cond
  | Not Cond
  | Between ValUnit Val Val
  | Eq ValUnit Val
  | Gt ValUnit Val
  | Lt ValUnit Val
  | Ge ValUnit Val
  | Le ValUnit Val
  | Ne ValUnit Val
  | In ValUnit Val
  | LIke ValUnit Val

data ColUnit
  = ColUnit
      { colUnitAggId :: AggType,
        colUnitColId :: ColumnId
      }
  | DistinctColUnit
      { distinctColUnitAggId :: AggType,
        distinctColUnitColdId :: ColumnId
      }

data OrderBy = OrderBy Order [ValUnit]

data Agg = Agg AggType ValUnit

data TableUnit = TableUnitSql Sql | Table TableId

data ValUnit
  = Column ColUnit
  | Minus ColUnit ColUnit
  | Plus ColUnit ColUnit
  | Times ColUnit ColUnit
  | Divide ColUnit ColUnit

data Val
  = Number Double
  | ValString String
  | ValSql Sql
  | ValColUnit ColUnit
  | Terminal

data AggType = NoneAggOp | Max | Min | Count | Sum | Avg

newtype ColumnId = ColumnId String

newtype TableId = TableId String

satisfyWordPiece :: MonadPlus b => (WordPiece -> Bool) -> Parser b Int WordPiece
satisfyWordPiece p = mfilter p (WordPiece . (t5Vocab Map.!) <$> token)

isWordPiece :: MonadPlus b => String -> Parser b Int WordPiece
isWordPiece s = satisfyWordPiece (== WordPiece s)

isSelect :: MonadPlus b => Parser b Int ()
isSelect = isWordPiece "\9601" >> isWordPiece "SEL" >> isWordPiece "ECT" $> ()

isComma :: MonadPlus b => Parser b Int ()
isComma = isWordPiece "," $> ()

isCount :: MonadPlus b => Parser b Int ()
isCount = isWordPiece "\9601CO" >> isWordPiece "UNT" $> ()

isFrom :: MonadPlus b => Parser b Int ()
isFrom = isWordPiece "\9601FROM" $> ()

isAs :: MonadPlus b => Parser b Int ()
isAs = isWordPiece "\9601AS" $> ()

isOn :: MonadPlus b => Parser b Int ()
isOn = isWordPiece "\9601ON" $> ()

isGroupBy :: MonadPlus b => Parser b Int ()
isGroupBy =
  isWordPiece "\9601G"
    >> isWordPiece "RO"
    >> isWordPiece "UP"
    >> isWordPiece "\9601B"
    >> isWordPiece "Y" $> ()

betweenParentheses :: MonadPlus b => Parser b Int a -> Parser b Int a
betweenParentheses = between (isWordPiece "(") (isWordPiece ")")

betweenOptionalParentheses :: MonadPlus b => Parser b Int a -> Parser b Int a
betweenOptionalParentheses p = betweenParentheses p <|> p

select :: MonadPlus b => Parser b Int Select
select = isSelect >> Select <$> sepBy agg isComma

agg :: MonadPlus b => Parser b Int Agg
agg = Agg <$> aggType <*> valUnit

aggType :: MonadPlus b => Parser b Int AggType
aggType = isCount $> Count <|> pure NoneAggOp

valUnit :: MonadPlus b => Parser b Int ValUnit
valUnit = betweenParentheses (choice [column, minus, plus, times, divide])
  where
    column = Column <$> colUnit
    minus = undefined
    plus = undefined
    times = undefined
    divide = undefined

colUnit :: MonadPlus b => Parser b Int ColUnit
colUnit = ColUnit <$> aggType <*> columnId

columnId :: MonadPlus b => Parser b Int ColumnId
columnId = ColumnId <$> string

string :: MonadPlus b => Parser b Int String
string =
  let f (c, _)
        | c == '▁' = True
        | otherwise = False
      p (WordPiece s) = maybe False f (uncons s)
      begin = satisfyWordPiece p
      append = satisfyWordPiece (not . p)
      strip ('▁' : s) = s
      strip s = s
      concat = strip . mconcat . (unWordPiece <$>)
   in concat <$> ((:) <$> begin <*> many append)

-- sql :: forall b. MonadPlus b => Parser b Int [Int]
-- sql =
--   let
--       is t = satisfy (== t)
--       dquote = is (WordPiece "\9601\"") $> ()
--       end = is (WordPiece "</s>") $> ()
--       in dquote >> manyTill token (dquote >> end)