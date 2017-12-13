{-# LANGUAGE DataKinds #-}

module Main where

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleRandom

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG
  let eigenvectors = tds_fromList [1, 0, 0.5, 1] :: TDS '[2,2]
  let eigenvalues  = tds_fromList [10, 30] :: TDS '[2]
  let mu           = tds_init 0.0 :: TDS '[2]
  result <- tds_mvn gen mu eigenvectors eigenvalues :: IO (TDS '[10, 2])
  tds_p result

main :: IO ()
main = do
  test_mvn
  putStrLn "Done"
