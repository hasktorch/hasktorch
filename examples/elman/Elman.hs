{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Elman where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import RecurrentLayer


data ElmanSpec = ElmanSpec { in_features :: Int, hidden_features :: Int }

data ElmanCell = ElmanCell {
    input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
}                            


instance RecurrentCell ElmanCell where

    nextState ElmanCell{..} input hidden = 
        gate input hidden Torch.Functions.tanh input_weight hidden_weight bias

    finalState layer input hidden =     
        let
        -- converting matrix into a list of tensors
        -- this hack stays until I can write a Foldable instance
        -- for a tensor
            inputAsList = [reshape (input @@ x) [1, 2] | x <- [0.. ((size input 0) - 1)]]
        in
            foldl (nextState layer) hidden inputAsList
        

instance Randomizable ElmanSpec ElmanCell where
    sample ElmanSpec{..} = do
      w_ih <- makeIndependent =<< randn' [in_features, hidden_features]
      w_hh <- makeIndependent =<< randn' [hidden_features, hidden_features]
      b <- makeIndependent =<< randn' [1, hidden_features]
      return $ ElmanCell w_ih w_hh b


instance Parameterized ElmanCell where
  flattenParameters ElmanCell{..} = [input_weight, hidden_weight, bias]
  replaceOwnParameters _ = do
    input_weight <- nextParameter
    hidden_weight <- nextParameter
    bias   <- nextParameter
    return $ ElmanCell{..}


instance Show ElmanCell where
    show ElmanCell{..} =
        (show input_weight) ++ "\n" ++
        (show hidden_weight) ++ "\n" ++
        (show bias) ++ "\n"

