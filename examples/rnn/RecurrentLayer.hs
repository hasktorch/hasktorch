{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RecurrentLayer where

import Control.Monad.State.Strict
import Data.List (foldl', intersperse, scanl')
import Torch

class RecurrentCell a where
  -- get the hidden state of the cell at the next timestep
  nextState :: a -> Tensor -> Tensor -> Tensor

  -- function to run the cell over multiple timesteps and get
  -- final hidden state
  finalState :: a -> Tensor -> Tensor -> Tensor
  finalState layer input hidden =
    let -- converting matrix into a list of tensors
        -- this hack stays until I can write a Foldable instance
        -- for a tensor
        inputAsList = [reshape [1, (size 1 input)] (input ! x) | x <- [0 .. ((size 0 input) - 1)]]
     in foldl (nextState layer) hidden inputAsList

{-
  TODO: there should also be a `forward` function here
  that uses the rnn forward functions from ATen
  but I'll implement that when I can make sense
  of the ATen function arguments -}

gate ::
  Tensor ->
  Tensor ->
  (Tensor -> Tensor) ->
  Parameter ->
  Parameter ->
  Parameter ->
  Tensor
gate input hidden nonLinearity inputWt hiddenWt biasWt =
  nonLinearity $ (mul input inputWt) + (mul hidden hiddenWt) + (toDependent biasWt)
  where
    mul features wts = transpose2D $ matmul (toDependent wts) (transpose2D features)
