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

seq_size = 5
num_features = 4


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

--    let input_tensor = fromNestedList $ map representation "hello"
    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    input_tensor <- randn' [seq_size, num_features]
    init_hidden <- randn' [1, num_features]
    expected_output <- randn' [1, num_features]

    -- randomly initialize a gate
    rnnLayer <- sample $ ElmanSpec { in_features = num_features, hidden_features = num_features }
    lstmLayer <- sample $ LSTMSpec num_features num_features
    gruLayer <- sample $ GRUSpec num_features num_features

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
