module Torch.Core.Tensor.Static.Random
  ( module Random
  ) where

import Torch.Core.Tensor.Dynamic.Random as Random

import Torch.Core.ByteTensor.Static.Random   ()
import Torch.Core.ShortTensor.Static.Random  ()
import Torch.Core.IntTensor.Static.Random    ()
import Torch.Core.LongTensor.Static.Random   ()
import Torch.Core.FloatTensor.Static.Random  ()
import Torch.Core.DoubleTensor.Static.Random ()

import Torch.Core.FloatTensor.Static.Random.Floating ()
import Torch.Core.DoubleTensor.Static.Random.Floating ()

{-
tds_mvn :: forall n p . (KnownNatDim n, KnownNatDim p) =>
  RandGen -> TDS '[p] -> TDS '[p,p] -> TDS '[p] -> IO (TDS '[n, p])
tds_mvn gen mu eigenvectors eigenvalues = do
  let offset = tds_expand mu :: TDS '[n, p]
  samps <- tds_normal gen 0.0 1.0 :: IO (TDS '[p, n])
  let result = tds_trans ((tds_trans eigenvectors)
                          !*! (tds_diag eigenvalues)
                          !*! eigenvectors
                          !*! samps) + offset
  pure result

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG
  let eigenvectors = tds_fromList [1, 1, 1, 1, 1, 1, 0, 0, 0] :: TDS '[3,3]
  tds_p eigenvectors
  let eigenvalues = tds_fromList [1, 1, 1] :: TDS '[3]
  tds_p eigenvalues
  let mu = tds_fromList [0.0, 0.0, 0.0] :: TDS '[3]
  result <- tds_mvn gen mu eigenvectors eigenvalues :: IO (TDS '[10, 3])
  tds_p result
-}
