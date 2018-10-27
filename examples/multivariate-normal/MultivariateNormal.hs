-------------------------------------------------------------------------------
--
-- This example illustrates sampling with hasktorch using a multivariate normal
-- sampler as an example.
--
-------------------------------------------------------------------------------

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad
import Torch.Double

-- | Create an RNG of Generator type, specify eigenvector/eigenvalues of the covariance matrix,
--   sample 10 samples of dimensions 2 (in the first case) and 3 (in the second case), print
--   their values.

main :: IO ()
main = do
  gen <- newRNG

  void $ do
    Just (evec :: Tensor '[2, 2]) <- fromList [1, 0, 0.5, 1]
    Just (eval :: Tensor '[2])    <- fromList [10, 30]
    runMultivariate gen evec eval

  void $ do
    Just (evec :: Tensor '[3, 3]) <- fromList [1, 1, 1, 1, 1, 1, 0, 0, 0]
    Just (eval :: Tensor '[3])    <- fromList [1, 1, 1]
    runMultivariate gen evec eval

  putStrLn "Done"

 where
  runMultivariate
    :: forall n
    .  KnownDim n
    => Generator
    -> Tensor '[n, n]
    -> Tensor '[n]
    -> IO ()
  runMultivariate gen eigenvector eigenvalue = do
    let mu :: Tensor '[n]     = constant 0
    result :: Tensor '[10, n] <- multivariate_normal gen mu eigenvector eigenvalue
    print result

