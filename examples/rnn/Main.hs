{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}

module Main where

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functional
import Torch.TensorOptions
import Torch.Autograd
import Torch.Optim
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
    => Tensor -- ^ input 
    -> Tensor -- ^ hidden state initial value
    -> Tensor -- ^ expected output
    -> a -- ^ model state
    -> Int -- ^ iteration
    -> IO (a) -- ^ new model state
run input_tensor init_hidden expected_output model i = do
    let output = finalState model input_tensor init_hidden
        loss = mse_loss output expected_output
    print loss 
    (newParam, _) <- runStep model GD loss 5e-2
    pure $ replaceParameters model newParam


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
