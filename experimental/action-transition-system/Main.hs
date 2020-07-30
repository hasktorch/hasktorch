module Main where

import Torch.Data.ActionTransitionSystem (testProgram)

main :: IO ()
main = testProgram 0.005 1000 1 1 "model.pt"
