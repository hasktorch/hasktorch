{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math
import Torch.Core.Tensor.Static.Random
import qualified Torch.Core.Random as RNG

type Tensor = DoubleTensor

test_mvn :: IO ()
test_mvn = do
  gen <- RNG.new
  eigenvectors  :: Tensor '[2,2] <- fromList [1, 0, 0.5, 1]
  eigenvalues   :: Tensor '[2]   <- fromList [10, 30]
  mu :: Tensor '[2] <- constant 0
  result :: Tensor '[10, 2] <- multivariate_normal gen mu eigenvectors eigenvalues
  printTensor result

  eigenvectors  :: Tensor '[3,3] <- fromList [1, 1, 1, 1, 1, 1, 0, 0, 0]
  eigenvalues   :: Tensor '[3]   <- fromList [1, 1, 1]
  mu :: Tensor '[3] <- constant 0
  result :: Tensor '[10, 3] <- multivariate_normal gen mu eigenvectors eigenvalues
  printTensor result

main :: IO ()
main = do
  test_mvn
  putStrLn "Done"
