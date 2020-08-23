{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.GRU where

import GHC.Generics
import Torch

data GRUSpec = GRUSpec {
    inputSize :: Int, 
    hiddenSize :: Int
} deriving (Eq, Show)

data GRUCell = GRUCell {
    weightsIH :: Parameter,
    weightsHH :: Parameter,
    biasIH :: Parameter,
    biasHH :: Parameter
} deriving (Generic, Show)

gruCellForward 
    :: GRUCell -- ^ cell parameters
    -> Tensor -- ^ input
    -> Tensor -- ^ hidden
    -> Tensor -- ^ output
gruCellForward GRUCell{..} input hidden =
    gruCell weightsIH' weightsHH' biasIH' biasHH' hidden input
    where
        weightsIH' = toDependent weightsIH
        weightsHH' = toDependent weightsHH
        biasIH' = toDependent biasIH
        biasHH' = toDependent biasHH

instance Parameterized GRUCell

instance Randomizable GRUSpec GRUCell where
  sample GRUSpec{..} = do
    weightsIH' <- makeIndependent =<< randIO' [3 * hiddenSize, inputSize]
    weightsHH' <- makeIndependent =<< randIO' [3 * hiddenSize, hiddenSize]
    biasIH' <- makeIndependent =<< randIO' [3 * hiddenSize]
    biasHH' <- makeIndependent =<< randIO' [3 * hiddenSize]
    pure $ GRUCell {
        weightsIH=weightsIH',
        weightsHH=weightsHH',
        biasIH=biasIH',
        biasHH=biasHH' }
    where
      scale = Prelude.sqrt $ 1.0 / fromIntegral hiddenSize :: Float
      initScale =  subScalar scale . mulScalar scale 
