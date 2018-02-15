{-# LANGUAGE DataKinds #-}

module Main where

import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Random

type Tensor = DoubleTensor

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG
  let eigenvectors = fromList [1, 0, 0.5, 1] :: Tensor '[2,2]
  let eigenvalues  = fromList [10, 30] :: Tensor '[2]
  let mu           = init 0.0 :: Tensor '[2]
  result <- tds_mvn gen mu eigenvectors eigenvalues :: IO (Tensor '[10, 2])
  tds_p result

main :: IO ()
main = do
  test_mvn
  putStrLn "Done"
