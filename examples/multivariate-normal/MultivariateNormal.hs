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

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG

  void $
    runMultivariate gen
      (fromList [1, 0, 0.5, 1] :: Maybe (Tensor '[2, 2]))
      (fromList [10, 30]       :: Maybe (Tensor '[2]))

  void $
    runMultivariate gen
      (fromList [1, 1, 1, 1, 1, 1, 0, 0, 0] :: Maybe (Tensor '[3, 3]))
      (fromList [1, 1, 1]                   :: Maybe (Tensor '[3]))

 where
  runMultivariate
    :: forall n
    .  KnownDim n
    => Generator
    -> Maybe (Tensor '[n, n])
    -> Maybe (Tensor '[n])
    -> IO ()
  runMultivariate gen (Just eigenvector) (Just eigenvalue) = do
    let mu :: Tensor '[n]     = constant 0
    result :: Tensor '[10, n] <- multivariate_normal gen mu eigenvector eigenvalue
    print result
  runMultivariate _ _ _ = pure ()

main :: IO ()
main = do
  test_mvn
  putStrLn "Done"

