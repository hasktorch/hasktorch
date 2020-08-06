{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.Elman where

import GHC.Generics
import Torch

data ElmanSpec = ElmanSpec { 
    inputSize :: Int,
    hiddenSize :: Int 
} deriving (Eq, Show)

data ElmanCell = ElmanCell {
    weightsIH :: Parameter,
    weightsHH :: Parameter,
    biasIH :: Parameter,
    biasHH :: Parameter
} deriving (Generic, Show)

elmanCellForward
    :: ElmanCell -- ^ cell parameters
    -> Tensor -- ^ input
    -> Tensor -- ^ hidden
    -> Tensor -- ^ output
elmanCellForward ElmanCell{..} input hidden =
    rnnReluCell weightsIH' weightsHH' biasIH' biasHH' hidden input
    where
        weightsIH' = toDependent weightsIH
        weightsHH' = toDependent weightsHH
        biasIH' = toDependent biasIH
        biasHH' = toDependent biasIH

instance Parameterized ElmanCell

instance Randomizable ElmanSpec ElmanCell where
    sample ElmanSpec{..} = do
      weightsIH <- makeIndependent =<< randnIO' [hiddenSize, inputSize]
      weightsHH <- makeIndependent =<< randnIO' [hiddenSize, hiddenSize]
      biasIH <- makeIndependent =<< randnIO' [hiddenSize]
      biasHH <- makeIndependent =<< randnIO' [hiddenSize]
      return $ ElmanCell weightsIH weightsHH biasIH biasHH
