{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad.State.Strict
import Control.Monad
import Data.List (foldl', intersperse, scanl')
import Elman
import GHC.Generics
import GRU
import LSTM
import RecurrentLayer
import Torch

num_iters = 10

num_timesteps = 3

run ::
  (RecurrentCell a, Parameterized a) =>
  -- | input
  Tensor ->
  -- | hidden state initial value
  Tensor ->
  -- | expected output
  Tensor ->
  -- | model state
  a ->
  -- | iteration
  Int ->
  -- | new model state
  IO (a)
run input_tensor init_hidden expected_output model i = do
  let output = finalState model input_tensor init_hidden
      loss = mseLoss expected_output output
  print loss
  (newParam, _) <- runStep model GD loss 5e-2
  pure newParam

main :: IO ()
main = do
  let foldLoop x count block = foldM block x [1 .. count]

  -- randomly initializing training values
  input_tensor <- randnIO' [num_timesteps, 2]
  init_hidden <- randnIO' [1, 2]
  expected_output <- randnIO' [1, 2]

  -- randomly initialize a gate
  rnnLayer <- sample $ ElmanSpec {in_features = 2, hidden_features = 2}
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
