{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Elman where

import Control.Monad.State.Strict
import Data.List (foldl', intersperse, scanl')
import RecurrentLayer
import Torch

data ElmanSpec = ElmanSpec {in_features :: Int, hidden_features :: Int}

data ElmanCell = ElmanCell
  { input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
  }

instance RecurrentCell ElmanCell where
  nextState ElmanCell {..} input hidden =
    gate input hidden Torch.tanh input_weight hidden_weight bias

instance Randomizable ElmanSpec ElmanCell where
  sample ElmanSpec {..} = do
    w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
    w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
    b <- makeIndependent =<< randnIO' [1, hidden_features]
    return $ ElmanCell w_ih w_hh b

instance Parameterized ElmanCell where
  flattenParameters ElmanCell {..} = [input_weight, hidden_weight, bias]
  _replaceParameters _ = do
    input_weight <- nextParameter
    hidden_weight <- nextParameter
    bias <- nextParameter
    return $ ElmanCell {..}

instance Show ElmanCell where
  show ElmanCell {..} =
    (show input_weight) ++ "\n"
      ++ (show hidden_weight)
      ++ "\n"
      ++ (show bias)
      ++ "\n"
