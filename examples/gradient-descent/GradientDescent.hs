{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Data.Monoid ((<>))
import Data.Function ((&))
import Data.Singletons
import GHC.TypeLits
import Lens.Micro

import Torch.Core.Tensor.Static.Double
import Torch.Core.Tensor.Static.DoubleMath
import Torch.Core.Tensor.Static.DoubleRandom
import Torch.Core.Random

import THDoubleTensor
import Foreign.ForeignPtr
import System.IO.Unsafe

type N = 1000 -- sample size
type NumP = 2
type P = '[1, 2]
seedVal = 223

genData :: TDS '[1,2] -> IO (TDS '[2, N], TDS '[N])
genData param = do
  rng <- newRNG
  manualSeed rng seedVal
  noise        :: TDS '[N] <- tds_normal rng 0.0 2.0
  predictorVal :: TDS '[N] <- tds_normal rng 0.0 10.0
  let x :: TDS '[2, N] =
        predictorVal
        & tds_cat (tds_init 1.0)
        & tds_resize
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

gradientDescent ::
  (TDS '[2, N], TDS '[N])
  -> TDS '[1, 2]
  -> Double
  -> Double
  -> [(TDS '[1, 2], Double, TDS '[1, 2])]
gradientDescent (x, y) param rate eps =
  if (tds_sumAll $ tds_abs g) < eps then
    []
  else
    [(param, j, g)] <>
    gradientDescent (x, y) (param - rate *^ g) rate eps
  where
    j = loss (x, y) param
    g = gradient (x, y) param

runN :: [(TDS '[1, 2], Double, TDS '[1, 2])] -> Int -> IO (TDS '[1,2])
runN lazyIters nIter = do
  let final = last $ take nIter lazyIters
      g = tds_sumAll . tds_abs . (^. _3) $ final
      j = (^. _2) final
      p = (^. _1) final
  putStrLn $ "Gradient magnitude after " <> show nIter <> " steps"
  print g
  putStrLn $ "Loss after " <> show nIter <> " steps"
  print j
  putStrLn $ "Parameter estimate after " <> show nIter <> " steps:"
  tds_p p
  pure p

runExample = do
  -- Generate data w/ ground truth params
  putStrLn "True parameters"
  let trueParam = tds_fromList [3.5, (-4.4)]
  tds_p trueParam
  dat <- genData trueParam

  -- Setup GD
  let p0 :: TDS '[1, 2] = tds_fromList [0.0, 0.0]
      iters = gradientDescent dat p0 0.001 0.001

  -- Results
  res <- runN iters 10000
  pure res

  -- peek at value w/o dispRaw pretty-printing
  putStrLn "Peek at raw pointer value of 2nd parameter:"
  let testVal = unsafePerformIO $ do
        withForeignPtr (tdsTensor res) (\pPtr -> pure $ c_THDoubleTensor_get2d pPtr 0 1)
  print testVal

main = do
  putStrLn "\nRun #1"
  runExample
  putStrLn "\nRun #2 using the same random seed"
  runExample
