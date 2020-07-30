module Main where

import Torch.Data.ActionTransitionSystem (testProgram)

main :: IO ()
main = testProgram 0.05 1000 1 1 "model.pt"
