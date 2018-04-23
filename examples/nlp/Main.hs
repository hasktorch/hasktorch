module Main where

import Torch.Cuda

type Tensor = DoubleTensor

main :: IO ()
main = do
  v_data :: Tensor '[2]    <- fromList [1..2]
  printTensor v_data
  v_data :: Tensor '[2, 3] <- fromList [1..2*3]
  printTensor v_data
  v_data :: Tensor '[2, 3, 2]    <- fromList [1..2*3*2]
  printTensor v_data
  v_data :: Tensor '[2, 3, 2, 3] <- fromList [1..2*3*2*3]
  printTensor v_data

  putStrLn "keep in mind:"
  v_data :: Tensor '[2, 3] <- fromList []
  printTensor v_data
  v_data :: Tensor '[2] <- fromList [1..5]
  printTensor v_data
