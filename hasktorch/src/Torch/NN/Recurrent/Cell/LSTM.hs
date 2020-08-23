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
    -> (Tensor, Tensor) -- ^ (hidden, cell)
    -> Tensor -- ^ input
    -> (Tensor, Tensor) -- ^ output (hidden, cell) 
lstmCellForward LSTMCell{..} hidden input =
    lstmCell weightsIH' weightsHH' biasIH' biasHH' hidden input
    where
        weightsIH' = toDependent weightsIH
        weightsHH' = toDependent weightsHH
        biasIH' = toDependent biasIH
        biasHH' = toDependent biasHH

instance Parameterized LSTMCell

instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
    -- x4 dimension calculations - see https://pytorch.org/docs/master/generated/torch.nn.LSTMCell.html
    weightsIH' <- makeIndependent =<< initScale <$> randIO' [4 * hiddenSize, inputSize]
    weightsHH' <- makeIndependent =<< initScale <$> randIO' [4 * hiddenSize, hiddenSize]
    biasIH' <- makeIndependent =<< initScale <$> randIO' [4 * hiddenSize]
    biasHH' <- makeIndependent =<< initScale <$> randIO' [4 * hiddenSize]
    pure $ LSTMCell {
        weightsIH=weightsIH',
        weightsHH=weightsHH',
        biasIH=biasIH',
        biasHH=biasHH' }
    where
      scale = Prelude.sqrt $ 1.0 / fromIntegral hiddenSize :: Float
      initScale =  subScalar scale . mulScalar scale 
