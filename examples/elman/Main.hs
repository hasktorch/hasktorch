{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

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

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------


main :: IO ()
main = do
    -- randomly initialize a gate
    (rnnLayer :: ElmanCell) <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }
    print rnnLayer
