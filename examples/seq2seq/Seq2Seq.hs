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
import GHC.Generics

import Control.Monad.State.Strict
import Data.List

import RecurrentLayer
import Elman
import LSTM
import GRU


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

--    if i `mod` 100 == 0
--    then print loss
--    else return ()

    let flat_parameters = flattenParameters model
    let gradients = grad loss flat_parameters

    -- new parameters returned by the SGD update functions
    new_flat_parameters <- mapM makeIndependent $ sgd 0.05 flat_parameters gradients

    -- return the new model state "to" the next iteration of foldLoop
    return $ replaceParameters model new_flat_parameters

{- generateExamples.hs, num->repr file, repr-> num file, this file -}
main = putStrLn "To be implemented"