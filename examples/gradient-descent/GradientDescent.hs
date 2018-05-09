{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
module Main where

import Data.Monoid ((<>))
import Control.Monad
import Lens.Micro

import Torch.Double hiding (N)
import qualified Torch.Double as Math
import qualified Torch.Core.Random as RNG

type N = 2000 -- sample size
type NumP = 2
type P = '[1, 2]
type Precision = Double

seedVal :: RNG.Seed
seedVal = 3141592653579

genData :: Tensor '[1,2] -> IO (Tensor '[2, N], Tensor '[N])
genData param = do
  gen <- newRNG
  RNG.manualSeed gen seedVal
  noise        :: Tensor '[N] <- normal gen 0 2
  predictorVal :: Tensor '[N] <- normal gen 0 10
  x :: Tensor '[2, N] <- (constant 1 >>= (\(o :: Tensor '[N]) -> predictorVal `cat1d` o) >>= resizeAs)
  y :: Tensor '[N]    <- (newTranspose2d (param !*! x) >>= resizeAs >>= Math.cadd noise 1)

  pure (x, y)

loss :: (Tensor '[2,N], Tensor '[N]) -> Tensor '[1, 2] -> IO Precision
loss (x, y) param = do
  x' <- (y -) <$> resizeAs (param !*! x)
  (realToFrac . Math.sumall) <$> Math.square x'


gradient
  :: forall n . (KnownNatDim n)
  => (Tensor '[2, n], Tensor '[n]) -> Tensor '[1, 2] -> IO (Tensor '[1, 2])
gradient (x, y) param = do
  y' :: Tensor '[1, n] <- resizeAs y
  x' :: Tensor '[n, 2] <- newTranspose2d x
  m  :: Tensor '[1, 2] <- resizeAs (err y' !*! x')
  pure $ (-2 / nsamp) *^ m

  where
    err :: Tensor '[1, n] -> Tensor '[1, n]
    err y' = y' - (param !*! x)

    nsamp :: Precision
    nsamp = realToFrac (natVal (Proxy :: Proxy n))

gradientDescent
  :: (Tensor '[2, N], Tensor '[N])
  -> Precision
  -> Precision
  -> Tensor '[1, 2]
  -> IO [(Tensor '[1, 2], Precision, Tensor '[1, 2])]
gradientDescent (x, y) rate eps = go 0 []
 where
  go :: Int -> [(Tensor '[1, 2], Precision, Tensor '[1, 2])] -> Tensor '[1, 2] -> IO [(Tensor '[1, 2], Precision, Tensor '[1, 2])]
  go i res param = do
    g <- gradient (x, y) param
    diff <- (realToFrac . Math.sumall) <$> Math.abs g
    if diff < eps
    then pure res
    else do
      j <- loss (x, y) param
      let param' = param ^-^ (g ^* rate)
      go (i+1) ((param, j, g):res) param'

runN :: [(Tensor '[1, 2], Precision, Tensor '[1, 2])] -> Int -> IO (Tensor '[1,2])
runN lazyIters nIter = do
  let final = last $ take nIter lazyIters
  g <- Math.sumall <$> Math.abs (final ^. _3)
  let j = (^. _2) final
  let p = (^. _1) final
  putStrLn $ "Gradient magnitude after " <> show nIter <> " steps"
  print g
  putStrLn $ "Loss after " <> show nIter <> " steps"
  print j
  putStrLn $ "Parameter estimate after " <> show nIter <> " steps:"
  print p
  pure p

runExample :: IO (Tensor '[1,2])
runExample = do
  -- Generate data w/ ground truth params
  putStrLn "True parameters"
  let Just trueParam = fromList [3.5, -4.4]
  print trueParam

  dat <- genData trueParam

  -- Setup GD
  let Just (p0 :: Tensor '[1, 2]) = fromList [0, 0]
  iters <- gradientDescent dat 0.0005 0.0001 p0

  -- Results
  x <- runN iters (fromIntegral (natVal (Proxy :: Proxy N)))
  pure x


  -- -- peek at value w/o dispRaw pretty-printing
  -- putStrLn "Peek at raw pointer value of 2nd parameter:"
  -- let testVal = unsafePerformIO $ do
  --       withForeignPtr (tdsTensor res) (\pPtr -> pure $ c_THTensor_get2d pPtr 0 1)
  -- print testVal

main :: IO ()
main = do
  putStrLn "\nRun #1"
  putStrLn "\nRun #2 using the same random seed"
  void runExample
