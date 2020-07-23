module Main where

import Torch.Data.ActionTransitionSystem (testProgram)

main :: IO ()
main = testProgram 0.001 100 1 1
