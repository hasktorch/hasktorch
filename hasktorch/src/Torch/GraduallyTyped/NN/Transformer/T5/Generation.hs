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

import Control.Monad.State (StateT (..), evalStateT, get, liftIO, put)
import Data.List (nub, sortOn)
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
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (expand)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), shape)
import qualified Torch.Tensor

data IsFinished = Finished | Unfinished

data Beam a b where
  Beam ::
    forall a b.
    { finished :: [Hypothesis 'Finished a b],
      unfinished :: [Hypothesis 'Unfinished a b]
    } ->
    Beam a b
  deriving (Show)

data Hypothesis (isFinished :: IsFinished) a b where
  InitialHypothesis ::
    forall a b.
    Hypothesis 'Unfinished a b
  UnfinishedHypothesis ::
    forall a b.
    { token :: a,
      score :: Float,
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

data SomeHypothesis a b = forall isFinished. SomeHypothesis {unSomeHypothesis :: Hypothesis isFinished a b}

beamSearch ::
  forall a b m.
  Monad m =>
  Int ->
  Int ->
  ([Hypothesis 'Unfinished a b] -> m [SomeHypothesis a b]) ->
  m [Beam a b]
beamSearch maxSteps beamSize cont = go maxSteps (Beam [] (replicate beamSize InitialHypothesis))
  where
    go !_ (Beam _ []) = pure []
    go n beam
      | n <= 0 = pure []
      | otherwise = do
        beam' <- beamSearchStep cont beam
        (beam' :) <$> go (n - 1) beam'

beamSearchStep ::
  forall a b m.
  Functor m =>
  ([Hypothesis 'Unfinished a b] -> m [SomeHypothesis a b]) ->
  Beam a b ->
  m (Beam a b)
beamSearchStep cont beam =
  let someHypotheses =
        take (length (unfinished beam))
          . reverse
          . sortOn @Float
            ( \case
                SomeHypothesis InitialHypothesis -> 0
                SomeHypothesis u@UnfinishedHypothesis {} -> score u
                SomeHypothesis f@FinishedHypothesis {} -> finalScore f
            )
          <$> (cont . unfinished $ beam)
   in foldl
        ( \(Beam fs us) someHypothesis ->
            case someHypothesis of
              SomeHypothesis u@InitialHypothesis -> Beam fs (u : us)
              SomeHypothesis u@UnfinishedHypothesis {} -> Beam fs (u : us)
              SomeHypothesis f@FinishedHypothesis {} -> Beam (f : fs) us
        )
        (Beam (finished beam) [])
        <$> someHypotheses

foo ::
  forall model input decoderInput encoderOutput encoderOutput' inputPaddingMask decoderOutput generator.
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
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size 15), 'Dim ( 'Name "*") ( 'Size 512)]),
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
  IO [Beam Int [Int]]
foo maxSteps beamSize model input g = do
  let cont :: [Hypothesis 'Unfinished Int [Int]] -> StateT (Maybe (encoderOutput, inputPaddingMask), generator) IO [SomeHypothesis Int [Int]]
      cont previousHypotheses = do
        let previousHypotheses' = nub previousHypotheses
        decoderInput :: decoderInput <- do
          let tokens =
                let go :: Hypothesis 'Unfinished Int [Int] -> [Int]
                    go InitialHypothesis = [] -- [t5PadTokenId]
                    go (UnfinishedHypothesis token' _ previousHypothesis') = token' : go previousHypothesis'
                 in reverse . go <$> previousHypotheses'
              batchSize = fromIntegral $ length previousHypotheses'
              seqSize =
                let go :: Hypothesis 'Unfinished Int [Int] -> Int
                    go InitialHypothesis = 0
                    go (UnfinishedHypothesis _ _ previousHypothesis') = 1 + go previousHypothesis'
                 in fromIntegral . maximum $ go <$> previousHypotheses'
          input <- mkT5Input @( 'Dim ( 'Name "*") 'UncheckedSize) @( 'Dim ( 'Name "*") 'UncheckedSize) batchSize seqSize tokens
          liftIO . print $ ((t5Vocab Map.!) <$>) <$> tokens
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
          let go :: Hypothesis 'Unfinished Int [Int] -> [Int]
              go InitialHypothesis = []
              go (UnfinishedHypothesis token' _ previousHypothesis') = token' : go previousHypothesis'
              finalValue = reverse $ token : go previousHypothesis
              go' :: Hypothesis 'Unfinished Int [Int] -> Float
              go' InitialHypothesis = 0
              go' (UnfinishedHypothesis _ previousScore _) = previousScore
              finalScore = logProb + go' previousHypothesis
           in SomeHypothesis (FinishedHypothesis token finalScore previousHypothesis finalValue)
        | otherwise =
          let go' :: Hypothesis 'Unfinished Int [Int] -> Float
              go' InitialHypothesis = 0
              go' (UnfinishedHypothesis _ previousScore _) = previousScore
              score = logProb + go' previousHypothesis
           in SomeHypothesis (UnfinishedHypothesis token score previousHypothesis)
  evalStateT (beamSearch maxSteps beamSize cont) (Nothing, g)

bar = do
  input <- do
    -- let tokens = [[13959, 1566, 12, 2968, 10, 6536, 43, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
    let tokens = [[13959, 1566, 12, 2968, 10, 148, 31, 60, 423, 13, 3, 7, 10536, 55, 1]]
    print $ ((t5Vocab Map.!) <$>) <$> tokens
    mkT5Input
      @( 'Dim ( 'Name "*") ( 'Size 1))
      @( 'Dim ( 'Name "*") ( 'Size 15))
      tokens
  model <-
    initialize
      @(T5Small 'WithLMHead ( 'Device 'CPU))
      "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt"
  g <- mkGenerator @( 'Device CPU) 0
  foo 20 10 model input g
