module Main where

import Torch
import Numeric.Backprop
import Data.Singletons.Prelude.Num


main :: IO ()
main = do
  putStrLn "let lenet begin!"

-- conv2d x y z = liftOp
