{-# LANGUAGE DataKinds #-}

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

type P = 10
type N = 100

-- TODO fill-in
main = do
  let x1 = tds_init 1.0 :: TDS '[5, 3]
      x2 = tds_init 2.0 :: TDS '[3, 4]
      x3 = (x1 !*! x2)
  tds_p x3
  putStrLn "Done"
  pure ()
