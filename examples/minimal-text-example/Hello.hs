{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad (when, foldM)
import Control.Monad.State.Strict
import Data.List
import Elman
import GHC.Generics
import GRU
import LSTM
import RecurrentLayer
import Torch

num_iters = 1000

num_features = 4

run ::
  (RecurrentCell a, Parameterized a) =>
  Tensor ->
  Tensor ->
  Tensor ->
  a ->
  Int ->
  IO (a)
run input_tensor init_hidden expected_output model i = do
  let output = finalState model input_tensor init_hidden
      loss = mseLoss expected_output output
  when (i `mod` 100 == 0) $ do
    print loss
  (newParam, _) <- runStep model GD loss 0.05
  return newParam

-- | convert a list to a one-dimensional tensor
fromList :: [Float] -> Tensor
fromList ls = asTensor ls

fromNestedList :: [[Float]] -> Tensor
fromNestedList ls = asTensor ls

-- | One-hot representation of the letter
repr :: Char -> [Float]
repr c = case c of
  'h' -> [1, 0, 0, 0]
  'e' -> [0, 1, 0, 0]
  'l' -> [0, 0, 1, 0]
  'o' -> [0, 0, 0, 1]

letter :: Int -> Char
letter index = case index of
  -1 -> '0'
  0 -> 'h'
  1 -> 'e'
  2 -> 'l'
  3 -> 'o'

letters :: [Tensor]
letters = map ((reshape [1, 4]) . (fromList . repr)) "helo"

getIndex :: Tensor -> Int
getIndex result = case index of
  Nothing -> -1
  Just x -> x
  where
    losses = map toDouble (map (flip mseLoss result) letters)
    min_loss = Prelude.minimum losses
    index = elemIndex min_loss losses

main :: IO ()
main = do
  --    let input_tensor = fromNestedList $ map representation "hello"
  let foldLoop x count block = foldM block x [1 .. count]

  let input_tensor = fromNestedList $ map repr "hell"

  -- randomly initialized hidden state
  init_hidden <- randnIO' [1, num_features]

  let expected_output = fromNestedList [repr 'o']

  -- randomly initialize a gate
  rnnLayer <- sample $ ElmanSpec {in_features = num_features, hidden_features = num_features}
  lstmLayer <- sample $ LSTMSpec num_features num_features
  gruLayer <- sample $ GRUSpec num_features num_features

  putStrLn "\nTraining Elman cell..."
  finalElman <- foldLoop rnnLayer num_iters (run input_tensor init_hidden expected_output)
  putStrLn "Testing Elman cell"
  let testElman = finalState finalElman input_tensor init_hidden
  putStrLn "Final letter after 'h-el-l-':"
  print $ letter $ getIndex testElman

  putStrLn "\nTraining LSTM cell..."
  finaLSTM <- foldLoop lstmLayer num_iters (run input_tensor init_hidden expected_output)
  putStrLn "Testing LSTM cell"
  let testLSTM = finalState finaLSTM input_tensor init_hidden
  putStrLn "Final letter after 'h-el-l-':"
  print $ letter $ getIndex testLSTM

  putStrLn "\nTraining GRU cell..."
  finalGRU <- foldLoop gruLayer num_iters (run input_tensor init_hidden expected_output)
  putStrLn "Testing GRU cell"
  let testGRU = finalState finalGRU input_tensor init_hidden
  putStrLn "Final letter after 'h-el-l-':"
  print $ letter $ getIndex testGRU

  return ()
