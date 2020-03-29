{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.LSTM where

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)
import GHC.Generics

import Torch
import Torch.NN.Recurrent.Cell.RecurrentLayer

data LSTMSpec = LSTMSpec { inf :: Int, hf :: Int}

data LSTMCell = LSTMCell {
	weightIH :: Parameter,
	weightHH :: Parameter,
	biasIH :: Parameter,
	biasHH :: Parameter
} deriving (Generic, Show)


runLSTMCell :: LSTMCell -> Tensor -> [Tensor] -> (Tensor, Tensor)
runLSTMCell LSTMCell{..} input hidden =
	lstmCell weightIH' weightHH' biasIH' biasHH' hidden input
	where
		weightIH' = toDependent weightIH
		weightHH' = toDependent weightHH
		biasIH' = toDependent biasIH
		biasHH' = toDependent biasHH

instance RecurrentCell LSTMCell where
  nextState cell input hidden = undefined
{-
    matmul og (Torch.tanh cNew)
    where
      og' = output_gate cell
      og = gate input hidden sigmoid
           (og' !! 0)
           (og' !! 1)
           (og' !! 2)
      cNew = newCellState cell input hidden
-}

instance Randomizable LSTMSpec LSTMCell where
  sample LSTMSpec{..} = do
	undefined
{-
      ig_ih <- makeIndependent =<< randnIO' [inf, hf]
      ig_hh <- makeIndependent =<< randnIO' [hf, hf]
      ig_b <- makeIndependent =<< randnIO' [1, hf]
      fg_ih <- makeIndependent =<< randnIO' [inf, hf]
      fg_hh <- makeIndependent =<< randnIO' [hf, hf]
      fg_b <- makeIndependent =<< randnIO' [1, hf]
      og_ih <- makeIndependent =<< randnIO' [inf, hf]
      og_hh <- makeIndependent =<< randnIO' [hf, hf]
      og_b <- makeIndependent =<< randnIO' [1, hf]
      hg_ih <- makeIndependent =<< randnIO' [inf, hf]
      hg_hh <- makeIndependent =<< randnIO' [hf, hf]
      hg_b <- makeIndependent =<< randnIO' [1, hf]
      let ig = [ig_ih, ig_hh, ig_b]
      let fg = [fg_ih, fg_hh, fg_b]
      let og = [og_ih, og_hh, og_b]
      let hg = [hg_ih, hg_hh, hg_b]
      c <- makeIndependent =<< randnIO' [hf, hf]
      return $ LSTMCell ig fg og hg c
-}
