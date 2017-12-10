{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.Function ((&))
import Torch.Core.Random (newRNG)
import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom

{- Types -}

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

{- Helper functions -}

seedVal :: Int
seedVal = 31415926535

genData :: TDS '[1,2] -> IO (TDS '[2, N], TDS '[N])
genData param = do
  gen <- newRNG
  manualSeed gen seedVal
  noise        :: TDS '[N] <- tds_normal gen 0.0 2.0
  predictorVal :: TDS '[N] <- tds_normal gen 0.0 10.0
  let x :: TDS '[2, N] =
        predictorVal
        & tds_cat (tds_init 1.0)
        & tds_resize
      y = (tds_trans (param !*! x))
          & tds_resize
          & (+) noise
  pure (x, y)

{- Main -}

main :: IO ()
main = do
  gen <- newRNG
  (x :: TDS '[N,P]) <- tds_uniform gen 0.0 20.0
  tds_p x
  let x1 = tds_init 1.0 :: TDS '[5, 3]
      x2 = tds_init 2.0 :: TDS '[3, 4]
      x3 = (x1 !*! x2)
  tds_p x3
  putStrLn "Done"
  pure ()
