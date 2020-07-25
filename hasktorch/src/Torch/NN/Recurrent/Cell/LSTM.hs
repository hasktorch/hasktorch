{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.LSTM where

import GHC.Generics
import Torch

data LSTMSpec = LSTMSpec {
    inputSize :: Int,
    hiddenSize :: Int
} deriving (Eq, Show)

data LSTMCell = LSTMCell {
    weightsIH :: Parameter,
    weightsHH :: Parameter,
    biasIH :: Parameter,
    biasHH :: Parameter
} deriving (Generic, Show)

lstmCellForward 
    :: LSTMCell -- ^ cell parameters
    -> Tensor -- ^ input
    -> (Tensor, Tensor) -- ^ (hidden, cell)
    -> (Tensor, Tensor) -- ^ output (hidden, cell) 
lstmCellForward LSTMCell{..} input hidden =
    lstmCell weightsIH' weightsHH' biasIH' biasHH' hidden input
    where
        weightsIH' = toDependent weightsIH
        weightsHH' = toDependent weightsHH
        biasIH' = toDependent biasIH
        biasHH' = toDependent biasHH

instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
    weightsIH' <- makeIndependent =<< randIO' [inputSize, hiddenSize]
    weightsHH' <- makeIndependent =<< randIO' [hiddenSize, hiddenSize]
    biasIH' <- makeIndependent =<< randIO' [hiddenSize]
    biasHH' <- makeIndependent =<< randIO' [hiddenSize]
    pure $ LSTMCell {
        weightsIH=weightsIH',
        weightsHH=weightsHH',
        biasIH=biasIH',
        biasHH=biasHH' }
