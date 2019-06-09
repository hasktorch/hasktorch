{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import Parameters
import LinearLayer
import RecurrentLayer
import MLP

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batch_size = 32
num_iters = 10000

model :: MLP -> Tensor -> Tensor
model params t = sigmoid (mlp params t)

sgd :: Tensor -> [Parameter] -> [Tensor] -> [Tensor]
sgd lr parameters gradients = zipWith (\p dp -> p - (lr * dp)) (map toDependent parameters) gradients

main :: IO ()
main = do
    init <- sample $ MLPSpec { feature_counts = [2, 1], nonlinearitySpec = Torch.Functions.tanh }
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }

    -- TODO: test if the cell function is correct
    inp <- randn' [2]
    hid <- randn' [2]
    let expected = inp

    let output = recurrent rnnLayer inp hid

    let loss = mse_loss output expected

    let flat_parameters = flattenParameters rnnLayer
    let gradients = grad loss flat_parameters
    print gradients
{-
        if i `mod` 100 == 0
          then do print loss
          else return ()

        new_flat_parameters <- mapM makeIndependent $ sgd 1.0 flat_parameters gradients
        return $ replaceParameters state $ new_flat_parameters
    return ()

  where
    foldLoop x count block = foldM block x [1..count]
-}
