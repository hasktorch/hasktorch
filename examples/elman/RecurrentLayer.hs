{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module RecurrentLayer where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)


class RecurrentCell a where
  -- get the hidden state of the cell at the next timestep
  nextState :: a -> Tensor -> Tensor -> Tensor
  -- function to run the cell over multiple timesteps and get
  -- final hidden state
  finalState :: a -> Tensor -> Tensor -> Tensor


{-
  TODO: there should also be a `forward` function here
  that uses the rnn forward functions from ATen
  but I'll implement that when I can make sense
  of the ATen function arguments -}


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



