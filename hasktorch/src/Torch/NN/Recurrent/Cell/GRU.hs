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

instance Randomizable GRUSpec GRUCell where
  sample GRUSpec{..} = do
    weightsIH' <- makeIndependent =<< randIO' [inputSize, hiddenSize]
    weightsHH' <- makeIndependent =<< randIO' [hiddenSize, hiddenSize]
    biasIH' <- makeIndependent =<< randIO' [hiddenSize]
    biasHH' <- makeIndependent =<< randIO' [hiddenSize]
    pure $ GRUCell {
        weightsIH=weightsIH',
        weightsHH=weightsHH',
        biasIH=biasIH',
        biasHH=biasHH' }
