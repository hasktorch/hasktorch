{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.GRU where

import GHC.Generics

import Torch
import Torch.NN.Recurrent.Cell.RecurrentLayer

data GRUSpec = GRUSpec { inf :: Int, hf :: Int}

data GRUCell = GRUCell {
	weightIH :: Parameter,
	weightHH :: Parameter,
	biasIH :: Parameter,
	biasHH :: Parameter
} deriving (Generic, Show)

gruCellForward :: GRUCell -> Tensor -> Tensor -> Tensor
gruCellForward GRUCell{..} input hidden =
	gruCell weightIH' weightHH' biasIH' biasHH' hidden input
	where
		weightIH' = toDependent weightIH
		weightHH' = toDependent weightHH
		biasIH' = toDependent biasIH
		biasHH' = toDependent biasHH

instance Randomizable GRUSpec GRUCell where
  sample GRUSpec{..} = do
    weightIH' <- makeIndependent =<< randIO' [inf, hf]
    weightHH' <- makeIndependent =<< randIO' [hf, hf]
    biasIH' <- makeIndependent =<< randIO' [hf]
    biasHH' <- makeIndependent =<< randIO' [hf]
    pure $ GRUCell {
        weightIH=weightIH',
        weightHH=weightHH',
        biasIH=biasIH',
        biasHH=biasHH' }
