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

genData :: RandGen -> TDS '[1,3] -> IO (TDS '[3, N], TDS '[N])
genData gen param = do
  manualSeed gen seedVal
  noise        :: TDS '[N] <- tds_normal gen 0.0 2.0
  x1 :: TDS '[N] <- tds_normal gen 0.0 10.0
  x2 :: TDS '[N] <- tds_normal gen 0.0 10.0
  let x0 = tds_init 1.0 :: TDS '[N]
  let x :: TDS '[3, N] =
        x1
        & tds_cat x2
        & tds_cat x0
        & tds_resize
      y = (tds_trans (param !*! x))
          & tds_resize
          & (+) noise
  pure (x, y)

genParam :: RandGen -> IO (TDS '[1, 3])
genParam gen = do
  let eigenvectors = tds_fromList [1, 0, 0, 0, 1, 1, 1, 0, 1] :: TDS '[3,3]
  let eigenvalues = tds_fromList [1, 1, 1]
  (predictorVal :: TDS '[1, 3]) <- tds_mvn gen
    (tds_init 0.0)
    eigenvectors
    eigenvalues
  putStrLn "Parameter values:"
  tds_p predictorVal
  pure predictorVal

{- Main -}

main :: IO ()
main = do
  gen <- newRNG
  param <- genParam gen
  (x, y) <- genData gen param
  putStrLn "x:"
  tds_p x
  putStrLn "x:"
  tds_p x
  putStrLn "y:"
  tds_p y
  putStrLn "y without noise:"
  tds_p $ param !*! x -- should be similar to y w/o noise
  putStrLn "Done"
  pure ()
