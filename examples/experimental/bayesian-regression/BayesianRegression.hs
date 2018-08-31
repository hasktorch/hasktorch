{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

--import Data.Function ((&))
import Torch.Double hiding (N)
import qualified Torch.Core.Random as RNG

{- Types -}

data Likelihood = Likelihood
  { l_mu    :: Double
  , l_sigma :: Double
  } deriving (Eq, Show)

data Prior = Prior
  { w_mu  :: Tensor '[P]
  , w_cov :: Tensor '[P, P]
  } deriving (Eq, Show)

newtype Samples = X
  { xMat :: Tensor '[N, P]
  } deriving (Eq, Show)

type P = 2
type N = 10

{- Helper functions -}

seedVal :: RNG.Seed
seedVal = 31415926535

genData :: RNG.Generator -> Tensor '[1,3] -> IO (Tensor '[3, N], Tensor '[N])
genData gen param = do
  RNG.manualSeed gen seedVal
  let
    Just pos2  = positive 2
    Just pos10 = positive 10
    x0 :: Tensor '[N] = constant 1

  noise :: Tensor '[N] <- normal gen 0.0 pos2
  x1    :: Tensor '[N] <- normal gen 0.0 pos10
  x2    :: Tensor '[N] <- normal gen 0.0 pos10
  let x :: Tensor '[3, N] = resizeAs (cat1d x0 (cat1d x1 x2))
  let y :: Tensor '[N] = noise ^+^ resizeAs (transpose2d (param !*! x))
  pure (x, y)

genParam :: RNG.Generator -> IO (Tensor '[1, 3])
genParam gen = do
  let Just (eigenvectors :: Tensor '[3,3]) = fromList [1, 0, 0, 0, 1, 1, 1, 0, 1]
  let Right (eigenvalues :: Tensor '[3]) = vector [1, 1, 1]
  let mu :: Tensor '[3] = constant 0
  predictorVal :: Tensor '[1, 3] <- multivariate_normal gen mu eigenvectors eigenvalues
  putStrLn "Parameter values:"
  print predictorVal
  pure predictorVal

{- Main -}

main :: IO ()
main = do
  gen <- RNG.newRNG
  param <- genParam gen
  (x, y) <- genData gen param
  putStrLn "x:"
  print x
  putStrLn "x:"
  print x
  putStrLn "y:"
  print y
  putStrLn "y without noise:"
  print $ param !*! x -- should be similar to y w/o noise
  putStrLn "Done"
  pure ()
