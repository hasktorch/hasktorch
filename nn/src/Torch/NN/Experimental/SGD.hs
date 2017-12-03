{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.NN.Experimental.SGD (
 ) where

import Data.Function ((&))
import GHC.TypeLits

import Pipes
import qualified Pipes.Prelude as P

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom
import Torch.Core.Random

-- toy test case

type N = 100 -- sample size
type B = 10  -- batch size

genData :: IO (TDS '[2, N], TDS '[N])
genData = do
  rng <- newRNG
  noise :: TDS '[N] <- tds_normal rng 0.0 2.0
  predictorVal :: TDS '[N]<- tds_normal rng 0.0 10.0
  let x :: TDS '[2, N] =
        predictorVal
        & tds_cat (tds_init 1.0)
        & tds_resize
      param :: TDS '[1, 2] = tds_fromList [3.5, 4.2]
      y = (tds_trans (param !*! x))
          & tds_resize
          & (+) noise
  pure (x, y)

loss :: (TDS '[2,N], TDS '[N]) -> TDS '[1, 2] -> Double
loss (x, y) param =
  (y ^-^ (tds_resize $ param !*! x))
  & flip tds_pow 2.0
  & tds_sumAll

gradient :: forall n . (KnownNat n) =>
  (TDS '[2,n], TDS '[n]) -> TDS '[1, 2] -> Double -> TDS '[2]
gradient (x, y) param rate =
  tds_sum (x !*! err) 1 False
  where
    err :: TDS '[n, 1] =
      tds_trans (rate *^ ((tds_resize y) ^-^ (param !*! x)))

main = do
  dat <- genData
  let p0 :: TDS '[1, 2] = tds_fromList [0.0, 0.0]
  let p1 :: TDS '[1, 2] = tds_fromList [1.0, 1.0]
  let p2 :: TDS '[1, 2] = tds_fromList [2.0, 2.0]
  let p :: TDS '[1, 2] = tds_fromList [3.5, 4.2]
  print $ loss dat p0
  print $ loss dat p1
  print $ loss dat p2
  print $ loss dat p

  let rate = 0.001
  tds_p $ gradient dat p0 rate
  tds_p $ gradient dat p1 rate
  tds_p $ gradient dat p2 rate
  tds_p $ gradient dat p rate
  putStrLn "Done"
