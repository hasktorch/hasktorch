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

import GenerateSamples

num_iters = 5
num_features = 8


run :: (RecurrentCell a, Parameterized a)
    => Tensor
    -> a
    -> (Tensor, Tensor)
    -> IO (a)
run init_hidden model io = do

    let input_tensor = fst io
    let expected_output = snd io
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


map' :: (Double -> Double) -> Tensor -> Tensor
map' f t = map'' [] t ((size t 1) - 1)
    where
        map'' acc tensor (-1) = asTensor $ reverse acc
        map'' acc tensor n = map'' ((f (toDouble (select tensor 1 n))) : acc) tensor (n - 1)

threshold' :: Double -> Double
threshold' a = if a > 0.5 then 1.0 else 0.0

eval :: RecurrentCell a
     => a   -- trained model
     -> Tensor  -- input
     -> Tensor  -- initial hidden tensor
     -> [Tensor]  -- accumulator
     -> Int       -- number of units left in sequence
     -> [Tensor]    -- output sequence
eval model input hidden acc 0 = reverse acc
eval model input hidden acc n = eval model input hidden (next : acc) (n - 1)
    where
        next = finalState model input hidden

main = do

    putStrLn "Generating examples..."
    samples <- generate 1
    print samples

    -- initialize encoder LSTM cell
    encoder <- sample $ LSTMSpec num_features num_features
    -- initialize decoder LSTM cell
    decoder <- sample $ LSTMSpec num_features num_features
    print "How do we train the encoder???"

