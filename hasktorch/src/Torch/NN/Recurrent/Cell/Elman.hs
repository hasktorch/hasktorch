{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.Elman where

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)
import GHC.Generics

import Torch
import Torch.NN.Recurrent.Cell.RecurrentLayer

data ElmanSpec = ElmanSpec { 
    in_features :: Int,
    hidden_features :: Int 
} deriving (Eq, Show)

data ElmanCell = ElmanCell {
    input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
} deriving (Generic, Show)

instance RecurrentCell ElmanCell where
    nextState ElmanCell{..} input hidden =
        gate input hidden Torch.tanh input_weight hidden_weight bias

instance Randomizable ElmanSpec ElmanCell where
    sample ElmanSpec{..} = do
      w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
      w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
      b <- makeIndependent =<< randnIO' [1, hidden_features]
      return $ ElmanCell w_ih w_hh b

{-
instance Parameterized ElmanCell where
  flattenParameters ElmanCell{..} = [input_weight, hidden_weight, bias]
  replaceOwnParameters _ = do
    input_weight <- nextParameter
    hidden_weight <- nextParameter
    bias   <- nextParameter
    return $ ElmanCell{..}
-}

{-
instance Show ElmanCell where
    show ElmanCell{..} =
        (show input_weight) ++ "\n" ++
        (show hidden_weight) ++ "\n" ++
        (show bias) ++ "\n"
-}