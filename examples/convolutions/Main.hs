module Main where

import qualified Conv1d
import qualified Conv2d
import qualified ReLU
import qualified MaxPooling
import qualified LeNet
import qualified DataLoader

main :: IO ()
main = do
  Conv1d.main
  Conv2d.main
  ReLU.main
  MaxPooling.main
  LeNet.main
  DataLoader.main

