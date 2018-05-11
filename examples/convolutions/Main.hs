module Main where

import qualified Conv1d
import qualified Conv2d
import qualified ReLU

main :: IO ()
main = do
  Conv1d.main
  Conv2d.main
  ReLU.main

