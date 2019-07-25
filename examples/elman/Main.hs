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



num_iters = 5
num_timesteps = 3


main :: IO ()
main = do

    -- randomly initialize the elman cell
    rnnLayer <- sample $ RecurrentSpec { in_features = 2, hidden_features = 2 }

    let foldLoop x count block = foldM block x [1..count]

    -- randomly initializing training values
    input_tensor <- randn' [num_timesteps, 2]
    init_hidden <- randn' [1, 2]
    expected_output <- randn' [1, 2]

    -- randomly initialize a gate
    rnnLayer <- sample $ ElmanSpec { in_features = 2, hidden_features = 2 }
    lstmLayer <- sample $ LSTMSpec 2 2
    gruLayer <- sample $ GRUSpec 2 2


    let foldLoop x count block = foldM block x [1..count]
    
    putStrLn "\nElman Cell Training Loop"
    -- training loop for elman cell
    foldLoop rnnLayer num_iters $ \model i -> do

        -- calculate output when RNN is run over timesteps
        let output = runOverTimesteps inp model init_hidden

        let output = finalState model input_tensor init_hidden
        let loss = mse_loss output expected_output

        print loss 

        let flat_parameters = flattenParameters model
        let gradients = grad loss flat_parameters
        

        -- new parameters returned by the SGD update functions
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        -- return the new model state "to" the next iteration of foldLoop
        return $ replaceParameters model new_flat_parameters

    putStrLn "\nLSTM Training Loop"
    -- training loop for LSTM cell
    foldLoop lstmLayer num_iters $ \model i -> do

        let output = finalState model input_tensor init_hidden
        let loss = mse_loss output expected_output

        print loss 

        let flat_parameters = flattenParameters model
        let gradients = grad loss flat_parameters
        
        -- new parameters returned by the SGD update functions
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        -- return the new model state "to" the next iteration of foldLoop
        return $ replaceParameters model new_flat_parameters

    putStrLn "\nGRU Training Loop"
    -- training loop for GRU cell
    foldLoop gruLayer num_iters $ \model i -> do

        let output = finalState model input_tensor init_hidden
        let loss = mse_loss output expected_output
        
        print loss 

        let flat_parameters = flattenParameters model
        let gradients = grad loss flat_parameters

        -- new parameters returned by the SGD update functions
        new_flat_parameters <- mapM makeIndependent $ sgd 5e-2 flat_parameters gradients

        return $ replaceParameters model new_flat_parameters

    return ()
