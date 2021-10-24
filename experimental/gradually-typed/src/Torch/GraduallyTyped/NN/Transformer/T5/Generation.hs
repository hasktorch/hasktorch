{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Generation where

import Control.Applicative (Alternative (..))
import Control.Monad (MonadPlus (..), guard)
import Control.Monad.Catch (MonadThrow)
import Control.Monad.State (MonadState (..), MonadTrans (..), StateT (..), evalStateT, gets, lift, modify)
import Control.Monad.Trans.Free (FreeF (..), FreeT (..), runFreeT)
import Data.Foldable (asum)
import Data.List (isInfixOf, nub, sortOn, uncons)
import qualified Data.Map as Map (Map, lookup)
import System.IO.Unsafe (unsafePerformIO)
import Text.Parser.Char (CharParsing (..), spaces)
import Text.Parser.Combinators (Parsing (..), between, manyTill)
import Text.Parser.Token (TokenParsing (..))
import Torch.Data.Parser (Parser, combine, isNotToken, isString, isToken, recurse, satisfy, scan, token)
import Torch.GraduallyTyped.DType (DType (..), DataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (fromStateDict), stateDictFromFile)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (logSoftmax)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (SimplifiedEncoderDecoderTransformerGenerationInput (..), SimplifiedEncoderDecoderTransformerInput (..), SimplifiedEncoderDecoderTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, mkT5Input, t5EOSTokenId)
import Torch.GraduallyTyped.NN.Transformer.T5.Small (t5SmallSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead))
import Torch.GraduallyTyped.NN.Type (SHasDropout (SWithDropout))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (sExpand)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison (Order (..), Sorted (..), sort)
import Torch.GraduallyTyped.Tensor.Type (SGetShape, Tensor (..))
import Torch.Language.SpiderSQL (SpiderSQL, spiderSQL)
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
  forall model input decoderInput encoderOutput encoderOutputShape encoderOutput' inputPaddingMask decoderOutput generatorDevice.
  ( HasForward
      model
      (SimplifiedEncoderDecoderTransformerInput input decoderInput)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
      generatorDevice,
    encoderOutput
      ~ Tensor
          ('Gradient 'WithGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          T5DataType
          encoderOutputShape,
    'UncheckedShape ~ BroadcastShapesF encoderOutputShape 'UncheckedShape,
    SGetShape encoderOutputShape,
    HasForward
      model
      (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput' inputPaddingMask)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput' decoderInput inputPaddingMask)
      generatorDevice,
    encoderOutput'
      ~ Tensor
          ('Gradient 'WithGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          T5DataType
          'UncheckedShape,
    decoderInput
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") 'UncheckedSize, 'Dim ('Name "*") 'UncheckedSize]),
    decoderOutput
      ~ Tensor
          ('Gradient 'WithGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          T5DataType
          'UncheckedShape,
    generatorDevice ~ 'Device 'CPU
  ) =>
  Int ->
  Int ->
  model ->
  input ->
  Generator generatorDevice ->
  IO [Beams Int [Int]]
runBeamSearch maxSteps beamSize model input g =
  evalStateT (beamSearch maxSteps beamSize cont) (Nothing, g)
  where
    cont :: [Hypothesis 'Unfinished Int [Int]] -> StateT (Maybe (encoderOutput, inputPaddingMask), Generator generatorDevice) IO [SomeHypothesis Int [Int]]
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
        mkT5Input (SName @"*" :&: SUncheckedSize batchSize) (SName @"*" :&: SUncheckedSize seqSize) (SDevice SCPU) tokens
      logProbs <- getLogProbs decoderInput
      pure $ zip previousHypotheses' logProbs >>= uncurry (\previousHypothesis -> zipWith (mkHypothesis previousHypothesis) [0, 1 ..] . last)
    getLogProbs :: decoderInput -> StateT (Maybe (encoderOutput, inputPaddingMask), Generator generatorDevice) IO [[[Float]]]
    getLogProbs decoderInput = do
      (maybeStuff, g) <- get
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput _ inputPaddingMask, g') <- case maybeStuff of
        Nothing -> forward model (SimplifiedEncoderDecoderTransformerInput input decoderInput) g
        Just (encoderOutput, inputPaddingMask) -> do
          -- decoderInputBatchDim : _ <- dims decoderInput
          decoderInputBatchDim <- undefined
          -- _encoderOutputBatchDim : encoderOutputDims <- dims encoderOutput
          encoderOutputDims <- undefined
          let encoderOutput' = sExpand (SUncheckedShape (decoderInputBatchDim : encoderOutputDims)) encoderOutput
          (SimplifiedEncoderDecoderTransformerOutput decoderOutput _ _ _, g') <- forward model (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput' inputPaddingMask) g
          pure (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask, g')
      put (Just (encoderOutput, inputPaddingMask), g')
      probs <- logSoftmax (SSelectDim $ SByIndex @2) decoderOutput
      case probs of
        UnsafeTensor t -> pure . Torch.Tensor.asValue . Torch.Tensor.Unsafe $ t
    mkHypothesis :: Hypothesis 'Unfinished Int [Int] -> Int -> Float -> SomeHypothesis Int [Int]
    mkHypothesis previousHypothesis token logProb
      | token == t5EOSTokenId =
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
    -- print $ ((t5Vocab Map.!) <$>) <$> tokens
    mkT5Input
      (SName @"*" :&: SSize @1)
      (SName @"*" :&: SSize @19)
      (SDevice SCPU)
      tokens
  stateDict <- stateDictFromFile "/tmp/t5-small-state-dict.pt"
  let spec = t5SmallSpec SWithLMHead (SGradient SWithGradient) (SDevice SCPU) SWithDropout
  model <- flip evalStateT stateDict $ fromStateDict spec mempty
  g <- sMkGenerator (SDevice SCPU) 0
  Beams finished _ <- last <$> runBeamSearch 50 1 model input g
  print $ finalValue <$> finished

-- let tmp = parseString @[] (transParser t5Vocab t5Text) . finalValue <$> finished
-- print tmp

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
      -- putStrLn $ "feed: " <> show (t5Vocab Map.! i)
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
              Free feed -> go cs (feed c)
       in Free feed'

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  Parsing (FreeT ((->) Char) (StateT [Int] b))
  where
  try = id
  (<?>) = const
  skipMany p = scan where scan = (p *> scan) <|> pure ()
  skipSome p = p *> skipMany p
  unexpected = const empty
  eof = undefined
  notFollowedBy = undefined

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  CharParsing (FreeT ((->) Char) (StateT [Int] b))
  where
  satisfy = Torch.Data.Parser.satisfy
  char = isToken
  notChar = isNotToken
  anyChar = Torch.Data.Parser.token
  string = isString

instance
  (Alternative b, Foldable b, MonadPlus b) =>
  TokenParsing (FreeT ((->) Char) (StateT [Int] b))

-- | Get continuations from model
getIs ::
  forall model input generatorDevice b decoderInput encoderOutput decoderOutput inputPaddingMask s.
  ( Alternative b,
    MonadThrow b,
    s ~ (Maybe (encoderOutput, inputPaddingMask), Generator generatorDevice),
    decoderInput
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ( 'Shape
              '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") 'UncheckedSize]
          ),
    decoderOutput
      ~ Tensor
          ('Gradient 'WithGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          T5DataType
          'UncheckedShape,
    HasForward
      model
      (SimplifiedEncoderDecoderTransformerInput input decoderInput)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
      generatorDevice,
    HasForward
      model
      (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
      generatorDevice
  ) =>
  Int ->
  model ->
  input ->
  StateT s (StateT [Int] b) Int
getIs n model input = do
  -- tokens <- reverse <$> lift get
  tokens <- do
    ts <- reverse <$> lift get
    let ts' = unsafePerformIO $ do
          -- putStrLn $ "tokens: " <> show ((t5Vocab Map.!) <$> ts)
          pure ts
    pure ts'
  decoderInput :: decoderInput <-
    mkT5Input
      (SName @"*" :&: SSize @1)
      (SName @"*" :&: SUncheckedSize (fromIntegral $ length tokens))
      (SDevice SCPU)
      [tokens]
  decoderOutput <- do
    (mTensors, g) <- get
    (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask, g') <-
      case mTensors of
        Nothing -> forward model (SimplifiedEncoderDecoderTransformerInput input decoderInput) g
        Just (encoderOutput, inputPaddingMask) ->
          forward model (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask) g
    put (Just (encoderOutput, inputPaddingMask), g')
    pure decoderOutput
  probs <- logSoftmax (SSelectDim $ SByIndex @2) decoderOutput
  case sort @('SelectDim ('ByIndex 2)) Descending probs of
    Sorted _ (UnsafeTensor indices) ->
      let indices' = take n . last . head . Torch.Tensor.asValue @[[[Int]]] . Torch.Tensor.Unsafe $ indices
       in lift . lift . asum $ pure <$> indices'

runParser ::
  forall model input generatorDevice b a.
  _ =>
  Int ->
  model ->
  input ->
  Generator generatorDevice ->
  Parser (StateT [Int] b) Int a ->
  b (a, [Int])
runParser n model input g =
  flip runStateT []
    . flip evalStateT (Nothing, g)
    . recurse (next (getIs n model input))

-- testParser = do
--   input <- do
--     let tokens = [[13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
--     -- let tokens = [[13959, 1566, 12, 2968, 10, 148, 31, 60, 423, 13, 3, 7, 10536, 55, 1]]
--     -- let tokens = [[13959, 1566, 12, 2968, 10, 3, 31, 7, 15, 3437, 3, 17, 4416, 4350, 6, 3476, 599, 1935, 61, 45, 4219, 38, 3, 17, 536, 1715, 14939, 38, 3, 17, 357, 30, 3, 17, 5411, 2427, 12925, 834, 23, 26, 3274, 3, 17, 4416, 2427, 12925, 834, 23, 26, 563, 57, 3, 17, 5411, 2427, 12925, 834, 23, 26, 31, 1]]
--     -- let tokens = [[13959, 1566, 12, 2968, 10, 96, 3, 23143, 14196, 332, 4416, 4350, 6, 2847, 17161, 599, 1935, 61, 21680, 4219, 6157, 332, 536, 3, 15355, 3162, 14939, 6157, 332, 357, 9191, 332, 5411, 2427, 12925, 834, 23, 26, 3274, 332, 4416, 2427, 12925, 834, 23, 26, 350, 4630, 6880, 272, 476, 3, 17, 5411, 2427, 12925, 834, 23, 26, 96, 1]]
--     print $ length <$> tokens
--     print $ ((t5Vocab Map.!) <$>) <$> tokens
--     mkT5Input
--       @('Dim ('Name "*") ('Size 1))
--       @('Dim ('Name "*") ('Size 61))
--       tokens
--   model <-
--     initialize
--       @(T5Small ('Device 'CPU))
--       "/Users/torsten.scholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
--   g <- mkGenerator @('Device CPU) 0
--   let outputs = runParser 5 model input g (transParser t5Vocab t5Test)
--   pure . fst $ observe outputs

-- | @t5Text@ parses a 'Char' sequence delimited by @</s>@ as a 'String'.
t5Text :: MonadPlus b => Parser b Char String
t5Text = manyTill Torch.Data.Parser.token (isString "</s>")

-- >>> head $ parseString @[] t5Test "Studien haben belegt, dass es gut ist, einen Hund zu haben</s>"
-- []
t5Test :: MonadPlus b => Parser b Char String
t5Test =
  notEnd 25
    `combine` isString "belegt"
    `combine` notEnd 5
    `combine` isString "dass es"
    `combine` notEnd 25
    `combine` isString "haben"
    `combine` isString "</s>"
  where
    notEnd n = scan f "" Torch.Data.Parser.token
      where
        f s a = case s ++ [a] of
          s'
            | "</s>" `isInfixOf` s' -> Nothing
            | length s' > n -> Nothing
            | otherwise -> Just s'

-- | @t5Sql@ parses a 'Char' sequence starting with @\"@ and ending with @\" </s>@
-- as 'SpiderSQL'.
t5Sql ::
  (TokenParsing (FreeT ((->) Char) b), MonadPlus b) =>
  Parser b Char SpiderSQL
t5Sql =
  let q = spaces *> char '\"' <* spaces
   in between q q spiderSQL <* isString "</s>"
