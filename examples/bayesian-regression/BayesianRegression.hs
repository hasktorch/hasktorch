{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Torch.Core.Random (newRNG)
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom (tds_uniform)

data Likelihood = Likelihood {
  l_mu :: Double,
  l_sigma :: Double
  } deriving (Eq, Show)

data Prior = Prior {
  w_mu :: TDS '[P],
  w_cov :: TDS '[P, P]
  } deriving (Eq, Show)

data Samples = X {
  xMat :: TDS '[N, P]
  } deriving (Eq, Show)

type P = 2
type N = 10

main :: IO ()
main = do
  rng <- newRNG
  (x :: TDS '[N,P]) <- tds_uniform rng 0.0 20.0
  tds_p x
  let x1 = tds_init 1.0 :: TDS '[5, 3]
      x2 = tds_init 2.0 :: TDS '[3, 4]
      x3 = (x1 !*! x2)
  tds_p x3
  putStrLn "Done"
  pure ()
