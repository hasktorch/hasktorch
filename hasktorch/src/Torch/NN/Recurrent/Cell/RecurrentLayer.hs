{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.NN.Recurrent.Cell.RecurrentLayer where

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Torch

class RecurrentCell a where
  nextState :: a -> Tensor -> Tensor -> Tensor
  finalState :: a -> Tensor -> Tensor -> Tensor
  finalState layer input hidden =
        let
            inputAsList = 
              [reshape [1, (size input 1)] (input @@ x) 
              | x <- [0.. ((size input 0) - 1)]]
        in
        foldl (nextState layer) hidden inputAsList

gate :: Tensor
     -> Tensor
     -> (Tensor -> Tensor)
     -> Parameter
     -> Parameter
     -> Parameter
     -> Tensor
gate input hidden nonLinearity inputWt hiddenWt biasWt =
    nonLinearity $ (mul input inputWt) + (mul hidden hiddenWt) + (toDependent biasWt)
    where
      mul features wts = transpose2D $ matmul (toDependent wts) (transpose2D features)