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
import Control.Monad (MonadPlus (..), guard, liftM, mfilter, replicateM)
import Control.Monad.Logic (observe)
import Control.Monad.State (MonadState (..), MonadTrans (..), StateT (..), evalStateT, gets, lift, modify)
import Control.Monad.Trans.Free (FreeF (..), FreeT (..), iterTM, runFreeT, transFreeT)
import Data.Char (isAlpha, isAlphaNum, isDigit, isSpace, toLower)
import Data.Foldable (asum)
import Data.Functor (($>))
import Data.Kind (Type)
import Data.List (isSuffixOf, nub, sortOn, uncons)
import qualified Data.Map as Map (Map, lookup, (!))
import Data.Maybe (maybeToList)
import System.IO.Unsafe (unsafePerformIO)
import Text.Read (readMaybe)
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
  let tmp = parseString @[] (transParser t5Vocab t5Text) . finalValue <$> finished
  print tmp

-- | @Parser b i a@ is a parser that consumes a stream of @i@ tokens and as a
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

-- | @transParser vocab p@ transforms a parser @p@ over characters 'Char'
-- into a parser over token indices 'Int' using the vocabulary @vocab@.
transParser :: MonadPlus b => Map.Map Int String -> Parser b Char a -> Parser b Int a
transParser vocab = FreeT . fmap (fmap (transParser vocab) . transFreeF) . runFreeT
  where
    transFreeF (Pure a) = Pure a
    transFreeF (Free feed) =
      let feed' i = do
            s <-
              let clean ('▁' : s) = ' ' : s
                  clean s = s
               in maybe empty (pure . clean) (Map.lookup i vocab)
            (c, cs) <- maybe empty pure (uncons s)
            go cs (feed c)
          go [] p = p
          go (c : cs) p = do
            val <- lift $ runFreeT p
            case val of
              Pure a -> pure a
              -- pure . unsafePerformIO $ do
              -- putStrLn $ "pure cs: " <> show cs
              -- pure a
              Free feed -> go cs (feed c)
       in -- unsafePerformIO $ do
          -- putStrLn $ "recurse c: " <> show c
          -- pure $ go cs (feed c)
          Free feed'

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
  let outputs = runParser model input g (transParser t5Vocab t5Sql)
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

-- | @combine p p'@ merges the results of @p@ and @p'@ using the 'Semigroup' instance.
combine :: (Applicative f, Semigroup a) => f a -> f a -> f a
combine = liftA2 (<>)

-- | @between open close p@ applies the parsers @open@, @p@, and @close@
-- in that order. Only the result of @p@ is returned, the results of @open@
-- and @close@ are discarded.
--
-- This combinator is useful for parsing expressions wrapped in parentheses,
-- for example.
between :: Applicative f => f a1 -> f a2 -> f a -> f a
between open close p = open *> p <* close

-- | @isString s@ is a simple parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. Otherwise, the parser fails.
isString :: MonadPlus b => String -> Parser b Char String
isString = traverse is

isNotString :: MonadPlus b => String -> Parser b Char String
isNotString = traverse isNot

string :: MonadPlus b => Parser b Char String
string = many token

string1 :: MonadPlus b => Parser b Char String
string1 = many1 token

-- | @t5Text@ parses a 'Char' sequence delimited by "</s>" as a 'String'.
t5Text :: MonadPlus b => Parser b Char String
t5Text = manyTill token (isString "</s>")

space :: MonadPlus b => Parser b Char String
space = many (satisfy isSpace)

space1 :: MonadPlus b => Parser b Char String
space1 = many1 (satisfy isSpace)

alpha1 :: MonadPlus b => Parser b Char String
alpha1 = many1 (satisfy isAlpha)

alphaNum1 :: MonadPlus b => Parser b Char String
alphaNum1 = many1 (satisfy isAlphaNum)

digits1 :: MonadPlus b => Parser b Char String
digits1 = many1 (satisfy isDigit)

-- newtype WordPiece = WordPiece {unWordPiece :: String} deriving (Eq, Ord, Show)

-- newtype Word = Word {unWord :: String} deriving (Eq, Ord, Show)

-- word :: forall b. MonadPlus b => Parser b Int Word
-- word =
--   let f (c, _)
--         | c == '▁' = True
--         | otherwise = False
--       p (WordPiece ",") = True
--       p (WordPiece ":") = True
--       p (WordPiece ".") = True
--       p (WordPiece ";") = True
--       p (WordPiece "!") = True
--       p (WordPiece "-") = True
--       p (WordPiece "?") = True
--       p (WordPiece "...") = True
--       p (WordPiece s) = maybe False f (uncons s)
--       p' (WordPiece "</s>") = False
--       p' _ = True
--       satisfy p = mfilter p (WordPiece . (t5Vocab Map.!) <$> token)
--       begin = satisfy ((&&) <$> p' <*> p)
--       append = satisfy ((&&) <$> p' <*> (not . p))
--       strip ('▁' : s) = s
--       strip s = s
--       concat = strip . mconcat . (unWordPiece <$>)
--    in Word . concat <$> ((:) <$> begin <*> many append)

-- words :: forall b. MonadPlus b => Parser b Int [Word]
-- words =
--   let satisfy p' = mfilter p' (WordPiece . (t5Vocab Map.!) <$> token)
--       is t = satisfy (== t)
--       end = is (WordPiece "</s>")
--       forceBeginning :: FreeT ((->) Int) b Word =
--         let rewrap (WordPiece ('▁' : s)) = Word s
--          in rewrap <$> is (WordPiece "\9601Wer")
--    in -- (:) <$> forceBeginning <*>
--       manyTill word end

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
  deriving (Show)

data Select
  = Select [Agg]
  | SelectDistinct [Agg]
  deriving (Show)

data From = From
  { fromTableUnits :: [TableUnit],
    fromCond :: Maybe Cond
  }
  deriving (Show)

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
  | Like ValUnit Val
  deriving (Show)

data ColUnit
  = ColUnit
      { colUnitAggId :: AggType,
        colUnitTable :: Maybe (Either TableId Alias),
        colUnitColId :: ColumnId
      }
  | DistinctColUnit
      { distinctColUnitAggId :: AggType,
        distinctColUnitTable :: Maybe (Either TableId Alias),
        distinctColUnitColdId :: ColumnId
      }
  deriving (Show)

data OrderBy = OrderBy Order [ValUnit] deriving (Show)

data Agg = Agg AggType ValUnit deriving (Show)

data TableUnit
  = TableUnitSql Sql (Maybe Alias)
  | Table TableId (Maybe Alias)
  deriving (Show)

data ValUnit
  = Column ColUnit
  | Minus ColUnit ColUnit
  | Plus ColUnit ColUnit
  | Times ColUnit ColUnit
  | Divide ColUnit ColUnit
  deriving (Show)

data Val
  = ValColUnit ColUnit
  | Number Double
  | ValString String
  | ValSql Sql
  | Terminal
  deriving (Show)

data AggType = NoneAggOp | Max | Min | Count | Sum | Avg deriving (Show)

newtype ColumnId = ColumnId String deriving (Show)

newtype TableId = TableId String deriving (Show)

newtype Alias = Alias String deriving (Show)

-- | @keyword k@ is a parser that consumes 'Char' tokens and yields them
-- if and only if they assemble the 'String' @s@. The parser is not sensitive to
-- letter casing.
--
-- >>> head $ parseString @[] (isKeyword "mykeyword") "MYKEYWORD"
-- ("MYKEYWORD","")
isKeyword :: MonadPlus b => String -> Parser b Char String
isKeyword = traverse (satisfy . ((. toLower) . (==) . toLower))

isSelect :: MonadPlus b => Parser b Char String
isSelect = isKeyword "select"

isDistinct :: MonadPlus b => Parser b Char String
isDistinct = isKeyword "distinct"

isStar :: MonadPlus b => Parser b Char String
isStar = pure <$> is '*'

isComma :: MonadPlus b => Parser b Char String
isComma = pure <$> is ','

isDot :: MonadPlus b => Parser b Char String
isDot = pure <$> is '.'

isEq :: MonadPlus b => Parser b Char String
isEq = pure <$> is '='

isCount :: MonadPlus b => Parser b Char String
isCount = isKeyword "count"

isFrom :: MonadPlus b => Parser b Char String
isFrom = isKeyword "from"

isJoin :: MonadPlus b => Parser b Char String
isJoin = isKeyword "join"

isAs :: MonadPlus b => Parser b Char String
isAs = isKeyword "as"

isOn :: MonadPlus b => Parser b Char String
isOn = isKeyword "on"

isWhere :: MonadPlus b => Parser b Char String
isWhere = isKeyword "where"

isGroupBy :: MonadPlus b => Parser b Char String
isGroupBy = isKeyword "group" `combine` space `combine` isKeyword "by"

isOrderBy :: MonadPlus b => Parser b Char String
isOrderBy = isKeyword "order" `combine` space `combine` isKeyword "by"

isHaving :: MonadPlus b => Parser b Char String
isHaving = isKeyword "having"

isLimit :: MonadPlus b => Parser b Char String
isLimit = isKeyword "limit"

betweenParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenParentheses = between (is '(') (is ')')

betweenOptionalParentheses :: MonadPlus b => Parser b Char a -> Parser b Char a
betweenOptionalParentheses p = betweenParentheses p <|> p

-- | 'Select' parser
--
-- >>> head $ parseString @[] select "select count table.*"
-- (Select [Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "table")), colUnitColId = ColumnId "*"}))],"")
select :: MonadPlus b => Parser b Char Select
select =
  space
    *> isSelect
    *> space
    *> ( (Select <$> sepBy agg isComma)
           <|> (isDistinct *> space *> (SelectDistinct <$> sepBy agg isComma))
       )
    <* space

agg :: MonadPlus b => Parser b Char Agg
agg = space *> (Agg <$> aggType <*> valUnit) <* space

aggType :: MonadPlus b => Parser b Char AggType
aggType = space *> ((isCount $> Count) <|> pure NoneAggOp) <* space

-- | 'ValUnit' parser
--
-- >>> head $ parseString @[] valUnit "t1.stadium_id"
-- (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}),"")
valUnit :: MonadPlus b => Parser b Char ValUnit
valUnit =
  space
    *> betweenOptionalParentheses
      ( space *> choice choices <* space
      )
    <* space
  where
    choices = [column] --, minus, plus, times, divide]
    column = Column <$> colUnit

colUnit :: MonadPlus b => Parser b Char ColUnit
colUnit =
  space
    *> ( ColUnit
           <$> aggType
           <*> maybeP (eitherP tableId alias <* isDot)
           <*> columnId
       )
    <* space

name :: MonadPlus b => Parser b Char String
name = many1 $ satisfy ((||) <$> isAlphaNum <*> (== '_'))

tableId :: MonadPlus b => Parser b Char TableId
tableId = TableId <$> name

alias :: MonadPlus b => Parser b Char Alias
alias = Alias <$> name

columnId :: MonadPlus b => Parser b Char ColumnId
columnId = ColumnId <$> (isStar <|> name)

-- | 'From' parser
--
-- >>> head $ parseString @[] from "FROM people AS t1 JOIN pets AS t2 ON t1.pet_id = t2.pet_id"
-- (From {fromTableUnits = [Table (TableId "people") (Just (Alias "t1")),Table (TableId "pets") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "pet_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "pet_id"})))},"")
from :: forall b. MonadPlus b => Parser b Char From
from = space *> isFrom *> space *> (uncurry mkFrom <$> p) <* space
  where
    p :: Parser b Char (TableUnit, [(TableUnit, Maybe Cond)])
    p =
      (,)
        <$> tableUnit
        <*> many
          ( space
              *> isJoin
              *> space
              *> ( (,)
                     <$> tableUnit
                     <*> maybeP
                       ( space
                           *> isOn
                           *> space
                           *> cond
                       )
                 )
          )
    mkFrom :: TableUnit -> [(TableUnit, Maybe Cond)] -> From
    mkFrom tu tus =
      From
        (tu : fmap fst tus)
        ( foldl
            ( \a b ->
                case (a, b) of
                  (Just c, Just c') -> Just (And c c')
                  (Just c, Nothing) -> Just c
                  (Nothing, Just c') -> Just c'
                  (Nothing, Nothing) -> Nothing
            )
            Nothing
            (fmap snd tus)
        )

tableUnit :: MonadPlus b => Parser b Char TableUnit
tableUnit = space *> (Table <$> tableId <*> maybeP (space *> isAs *> space *> alias)) <* space

-- | Condition parser
-- >>> head $ parseString @[] cond "t1.stadium_id = t2.stadium_id"
-- (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "stadium_id"})),"")
cond :: MonadPlus b => Parser b Char Cond
cond = space *> (Eq <$> valUnit <*> (space *> isEq *> space *> val)) <* space

val :: MonadPlus b => Parser b Char Val
val = space *> choice choices <* space
  where
    choices = [valColUnit, valString, terminal]
    -- choices = [valColUnit, number, valString, valSql, terminal]
    valColUnit = ValColUnit <$> colUnit
    -- number = undefined
    valString = ValString <$> string
    -- valSql = undefined
    terminal = pure Terminal

whereCond :: MonadPlus b => Parser b Char (Maybe Cond)
whereCond = space *> maybeP (isWhere *> space *> cond) <* space

groupBy :: MonadPlus b => Parser b Char [ColUnit]
groupBy = space *> (maybeToList <$> maybeP (isGroupBy *> space *> colUnit)) <* space

orderBy :: forall b. MonadPlus b => Parser b Char (Maybe OrderBy)
orderBy =
  space *> maybeP (isOrderBy *> space *> p) <* space
  where
    p :: Parser b Char OrderBy
    p = pure $ OrderBy Ascending []

havingCond :: MonadPlus b => Parser b Char (Maybe Cond)
havingCond = space *> maybeP (isHaving *> space *> cond) <* space

limit :: MonadPlus b => Parser b Char (Maybe Int)
limit = space *> maybeP (isLimit *> space *> digits1 >>= maybe empty pure . readMaybe) <* space

-- | 'Sql' parser
--
-- >>> head $ parseString @[] sql "select T2.name, count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id"
-- (Sql {sqlSelect = Select [Agg NoneAggOp (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "T2")), colUnitColId = ColumnId "name"})),Agg Count (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Nothing, colUnitColId = ColumnId "*"}))], sqlFrom = From {fromTableUnits = [Table (TableId "concert") (Just (Alias "t1")),Table (TableId "stadium") (Just (Alias "t2"))], fromCond = Just (Eq (Column (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"})) (ValColUnit (ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t2")), colUnitColId = ColumnId "stadium_id"})))}, sqlWhere = Nothing, sqlGroupBy = [ColUnit {colUnitAggId = NoneAggOp, colUnitTable = Just (Left (TableId "t1")), colUnitColId = ColumnId "stadium_id"}], sqlOrderBy = Nothing, sqlHaving = Nothing, sqlLimit = Nothing, sqlIntersect = Nothing, sqlExcept = Nothing, sqlUnion = Nothing},"")
sql :: MonadPlus b => Parser b Char Sql
sql =
  Sql
    <$> select
      <*> from
      <*> whereCond
      <*> groupBy
      <*> orderBy
      <*> havingCond
      <*> limit
      <*> pure Nothing
      <*> pure Nothing
      <*> pure Nothing

t5Sql :: MonadPlus b => Parser b Char Sql
t5Sql =
  let q = space *> is '\"' <* space
   in between q q sql <* isString "</s>"