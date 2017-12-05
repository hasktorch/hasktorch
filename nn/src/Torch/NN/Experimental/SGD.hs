{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.NN.Experimental.SGD (
 ) where

import Data.Monoid ((<>))
import Data.Function ((&))
import Data.Singletons
import GHC.TypeLits
import Lens.Micro

import Pipes hiding (Proxy)
import qualified Pipes.Prelude as P

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom
import Torch.Core.Random

-- Toy test case

type N = 100000 -- sample size
type B = 10  -- batch size
type P = '[1, 2]

genData :: IO (TDS '[2, N], TDS '[N])
genData = do
  rng <- newRNG
  noise        :: TDS '[N] <- tds_normal rng 0.0 2.0
  predictorVal :: TDS '[N] <- tds_normal rng 0.0 10.0
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
  & tds_square
  & tds_sumAll

gradient :: forall n . (KnownNat n) =>
  (TDS '[2, n], TDS '[n]) -> TDS '[1, 2] -> TDS '[1, 2]
gradient (x, y) param =
  ((-2.0) / nsamp) *^ (err !*! tds_trans x)
  where
    nsamp = realToFrac $ natVal (Proxy :: Proxy n)
    err :: TDS '[1,n] = tds_resize y - (param !*! x)

gradientDescent (x, y) param rate eps =
  if (tds_sumAll $ tds_abs g) < eps then []
  else
    ([(param, j, g)] <>
     gradientDescent (x, y) (param - rate *^ g) rate eps
    )
  where
    j = loss (x, y) param
    g = gradient (x, y) param

p0 :: TDS '[1, 2] = tds_fromList [0.0, 0.0]

diff [] = []
diff ls = zipWith (-) (tail ls) ls

main = do
  dat <- genData
  let result = gradientDescent dat p0 0.001 0.1
  let r = tds_sumAll . tds_abs . (^. _3) . last $ take 10000 result
  print r


