{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad
import Torch.Double

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG

  void . join $
    runMultivariate gen
      <$> (fromList [1, 0, 0.5, 1] :: IO (Tensor '[2, 2]))
      <*> (fromList [10, 30]       :: IO (Tensor '[2]))

  void . join $
    runMultivariate gen
      <$> (fromList [1, 1, 1, 1, 1, 1, 0, 0, 0] :: IO (Tensor '[3, 3]))
      <*> (fromList [1, 1, 1]                   :: IO (Tensor '[3]))

 where
  runMultivariate :: KnownNatDim n => Generator -> Tensor '[n, n] -> Tensor '[n] -> IO ()
  runMultivariate gen eigenvector eigenvalue = do
    mu     :: Tensor '[n]     <- constant 0
    result :: Tensor '[10, n] <- multivariate_normal gen mu eigenvector eigenvalue
    print result

main :: IO ()
main = do
  test_mvn
  putStrLn "Done"

