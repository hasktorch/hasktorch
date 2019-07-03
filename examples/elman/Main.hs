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
import RecurrentLayer
import Utils

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

num_iters = 10000
num_timesteps = 3


main :: IO ()
main = do
    -- randomly initialize the elman cell
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2, nonlinearitySpec = Torch.Functions.tanh }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    inp <- randn' [num_timesteps, 2]
    init_hidden <- randn' [1, 2]
    expected_output <- randn' [1, 2]

    -- training loop
    foldLoop rnnLayer num_iters $ \model i -> do

        -- calculate output when RNN is run over timesteps
        let output = runOverTimesteps inp model num_timesteps init_hidden

        let loss = mse_loss output expected_output

        -- "flatten" parameters into a single list to make it
        -- easier for libtorch grad to work with
        let flat_parameters = flattenParameters model

        -- gradients using libtorch grad functions
        let gradients = grad loss flat_parameters

        -- print loss every 100 iterations
        -- if the RNN is working, loss should reduce
        if i `mod` 100 == 0
            then do putStrLn $ show loss
          else return ()

        -- new parameters returned by the SGD update functions
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        -- return the new model state "to" the next iteration of foldLoop
        return $ replaceParameters model $ new_flat_parameters


    return ()
