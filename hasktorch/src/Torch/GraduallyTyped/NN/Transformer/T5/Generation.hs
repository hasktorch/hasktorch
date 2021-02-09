{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Generation where

import Control.Applicative (Alternative (many, (<|>)), Applicative (liftA2), empty)
import Control.Monad (MonadPlus, guard, mfilter)
import Control.Monad.State (MonadState (..), StateT (..), evalStateT)
import Control.Monad.Trans.Free (FreeF (Free, Pure), FreeT (FreeT), iterTM)
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
foo maxSteps beamSize model input g = do
  let cont :: [Hypothesis 'Unfinished Int [Int]] -> StateT (Maybe (encoderOutput, inputPaddingMask), generator) IO [SomeHypothesis Int [Int]]
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
          input <- mkT5Input @( 'Dim ( 'Name "*") 'UncheckedSize) @( 'Dim ( 'Name "*") 'UncheckedSize) batchSize seqSize tokens
          -- liftIO . print $ ((t5Vocab Map.!) <$>) <$> tokens
          -- liftIO . print $ shape input
          -- liftIO . print $ input
          pure input
        logProbs :: [[[Float]]] <- do
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
        pure $ zip previousHypotheses' logProbs >>= uncurry (\previousHypothesis -> zipWith (mkHypothesis previousHypothesis) [0, 1 ..] . last)
      mkHypothesis :: Hypothesis 'Unfinished Int [Int] -> Int -> Float -> SomeHypothesis Int [Int]
      mkHypothesis previousHypothesis token logProb
        | token == t5EosTokenId =
          let finalValue = reverse $ token : getTokens previousHypothesis
              finalScore = logProb + getScore previousHypothesis
           in SomeHypothesis (FinishedHypothesis token finalScore previousHypothesis finalValue)
        | otherwise =
          let score = logProb + getScore previousHypothesis
           in SomeHypothesis (UnfinishedHypothesis token score previousHypothesis)
  evalStateT (beamSearch maxSteps beamSize cont) (Nothing, g)

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
  let tmp = fmap (WordPiece . (t5Vocab Map.!)) . finalValue <$> finished
      tmp' = parseString @[] words <$> tmp
  print tmp'

newtype WordPiece = WordPiece {unWordPiece :: String} deriving (Eq, Ord, Show)

newtype Word = Word {unWord :: String} deriving (Eq, Ord, Show)

word :: forall b. MonadPlus b => Parser b WordPiece Word
word =
  let f (c, _)
        | c == '▁' = True
        | otherwise = False
      p (WordPiece ",") = True
      p (WordPiece s) = maybe False f (uncons s)
      begin = predicate p
      append = predicate (not . p)
      strip ('▁' : s) = s
      strip s = s
      concat = strip . mconcat . (unWordPiece <$>)
   in Word . concat <$> liftA2 (:) begin (many append)

words :: forall b. MonadPlus b => Parser b WordPiece [Word]
words =
  let end = is (WordPiece "</s>")
   in manyTill word end

type Parser (b :: Type -> Type) (t :: Type) (a :: Type) = FreeT ((->) t) b a

parseStream :: Monad b => (s -> b (t, s)) -> Parser b t a -> s -> b (a, s)
parseStream next = runStateT . iterTM (StateT next >>=)

-- | Parse a string. When the end of the string is encountered, 'empty' is
-- yielded into the non-determinism monad.
parseString :: forall b t a. MonadPlus b => Parser b t a -> [t] -> b (a, [t])
parseString = parseStream (maybe empty pure . uncons)

token :: forall b t. Applicative b => Parser b t t
token = FreeT . pure . Free $ FreeT . pure . Pure

predicate :: forall b t. (MonadPlus b) => (t -> Bool) -> Parser b t t
predicate p = mfilter p token

is :: forall b t. (MonadPlus b, Eq t) => t -> Parser b t t
is t = predicate (== t)

isNot :: forall b t. (MonadPlus b, Eq t) => t -> Parser b t t
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
