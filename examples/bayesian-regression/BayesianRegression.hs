{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

--import Data.Function ((&))
import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math
import Torch.Core.Tensor.Static.Random
import qualified Torch.Core.Random as RNG

{- Types -}
type Tensor = DoubleTensor

data Likelihood = Likelihood {
  l_mu :: Double,
  l_sigma :: Double
  } deriving (Eq, Show)

data Prior = Prior {
  w_mu :: Tensor '[P],
  w_cov :: Tensor '[P, P]
  } deriving (Eq, Show)

data Samples = X {
  xMat :: Tensor '[N, P]
  } deriving (Eq, Show)

type P = 2
type N = 10

{- Helper functions -}

seedVal :: RNG.Seed
seedVal = 31415926535

genData :: RNG.Generator -> Tensor '[1,3] -> IO (Tensor '[3, N], Tensor '[N])
genData gen param = do
  RNG.manualSeed gen seedVal
  noise :: Tensor '[N] <- normal gen 0.0 2.0
  x1    :: Tensor '[N] <- normal gen 0.0 10.0
  x2    :: Tensor '[N] <- normal gen 0.0 10.0
  x0    :: Tensor '[N] <- constant 1
  x     :: Tensor '[3, N] <- (cat1d x1 x2) >>= cat1d x0 >>= resizeAs
  y     :: Tensor '[N] <- (^+^) noise <$> (newTranspose2d (param !*! x) >>= resizeAs)
  pure (x, y)

genParam :: RNG.Generator -> IO (Tensor '[1, 3])
genParam gen = do
  eigenvectors :: Tensor '[3,3] <- fromList [1, 0, 0, 0, 1, 1, 1, 0, 1]
  eigenvalues :: Tensor '[3] <- fromList1d [1, 1, 1]
  mu :: Tensor '[3] <- constant 0
  predictorVal :: Tensor '[1, 3] <- multivariate_normal gen mu eigenvectors eigenvalues
  putStrLn "Parameter values:"
  printTensor predictorVal
  pure predictorVal

{- Main -}

main :: IO ()
main = do
  gen <- RNG.new
  param <- genParam gen
  (x, y) <- genData gen param
  putStrLn "x:"
  printTensor x
  putStrLn "x:"
  printTensor x
  putStrLn "y:"
  printTensor y
  putStrLn "y without noise:"
  printTensor $ param !*! x -- should be similar to y w/o noise
  putStrLn "Done"
  pure ()
