{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Torch.Autograd
import Torch.NN
import GHC.Generics

import Control.Monad.State.Strict
import Data.List (foldl', scanl', intersperse)

import RecurrentLayer
import Elman
import LSTM
import GRU


num_iters = 10
num_timesteps = 3

run :: (RecurrentCell a, Parameterized a) 
    => Tensor
    -> Tensor
    -> Tensor
    -> a
    -> Int 
    -> IO (a)
run input_tensor init_hidden expected_output model i = do
    
    let output = finalState model input_tensor init_hidden
    let loss = mse_loss output expected_output

    print loss 

    let flat_parameters = flattenParameters model
    let gradients = grad loss flat_parameters
        

    -- new parameters returned by the SGD update functions
    new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

    -- return the new model state "to" the next iteration of foldLoop
    return $ replaceParameters model new_flat_parameters


main :: IO ()
main = do

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    input_tensor <- randn' [num_timesteps, 2]
    init_hidden <- randn' [1, 2]
    expected_output <- randn' [1, 2]

    -- randomly initialize a gate
    rnnLayer <- sample $ ElmanSpec { in_features = 2, hidden_features = 2 }
    lstmLayer <- sample $ LSTMSpec 2 2
    gruLayer <- sample $ GRUSpec 2 2

    putStrLn "\nElman Cell Training Loop"
    -- training loop for elman cell
    foldLoop rnnLayer num_iters (run input_tensor init_hidden expected_output) 

    putStrLn "\nLSTM Training Loop"
    -- training loop for LSTM cell
    foldLoop lstmLayer num_iters (run input_tensor init_hidden expected_output)

    putStrLn "\nGRU Training Loop"
    -- training loop for GRU cell
    foldLoop gruLayer num_iters (run input_tensor init_hidden expected_output)

    return ()
