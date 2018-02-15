{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Data.Monoid ((<>))
import Data.Function ((&))
import Data.Singletons
import GHC.TypeLits
import Lens.Micro
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Dim hiding (N)
import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Static.Math as Math
import Torch.Core.Tensor.Static.Random
import qualified Torch.Core.Random as RNG

type N = 2000 -- sample size
type NumP = 2
type P = '[1, 2]
type Tensor = DoubleTensor

seedVal :: RNG.Seed
seedVal = 3141592653579

genData :: Tensor '[1,2] -> IO (Tensor '[2, N], Tensor '[N])
genData param = do
  gen <- RNG.new
  RNG.manualSeed gen seedVal
  noise        :: Tensor '[N] <- normal gen 0.0 2.0
  predictorVal :: Tensor '[N] <- normal gen 0.0 10.0
  x :: Tensor '[2, N] <- (constant 1 >>= cat1d predictorVal >>= resizeAs)
  y :: Tensor '[N]    <- (^+^ noise) <$> (newTranspose2d (param !*! x) >>= resizeAs)
  pure (x, y)

loss :: (Tensor '[2,N], Tensor '[N]) -> Tensor '[1, 2] -> IO Double
loss (x, y) param = do
  x <- (y ^-^) <$> resizeAs (param !*! x)
  sumall =<< Math.square x

gradient
  :: forall n . (KnownNatDim n)
  => (Tensor '[2, n], Tensor '[n]) -> Tensor '[1, 2] -> IO (Tensor '[1, 2])
gradient (x, y) param = do
  y' :: Tensor '[1, n] <- resizeAs y
  x' :: Tensor '[n, 2] <- newTranspose2d x
  pure $ ((-2) / nsamp) *^ (err y' !*! x')
  where
    err :: Tensor '[1, n] -> Tensor '[1, n]
    err y' = y' ^-^ (param !*! x)

    nsamp :: Double
    nsamp = (realToFrac $ natVal (Proxy :: Proxy n))


gradientDescent ::
  (Tensor '[2, N], Tensor '[N])
  -> Tensor '[1, 2]
  -> Double
  -> Double
  -> IO [(Tensor '[1, 2], Double, Tensor '[1, 2])]
gradientDescent (x, y) param rate eps = do
  g <- gradient (x, y) param
  d <- sumall =<< Math.abs g
  if d < eps
  then pure []
  else do
    j <- loss (x, y) param
    nxt <- gradientDescent (x, y) (param - rate *^ g) rate eps
    pure $ (param, j, g):nxt

runN :: [(Tensor '[1, 2], Double, Tensor '[1, 2])] -> Int -> IO (Tensor '[1,2])
runN lazyIters nIter = do
  let final = last $ take nIter lazyIters
  g <- sumall =<< Math.abs (final ^. _3)
  let j = (^. _2) final
  let p = (^. _1) final
  putStrLn $ "Gradient magnitude after " <> show nIter <> " steps"
  print g
  putStrLn $ "Loss after " <> show nIter <> " steps"
  print j
  putStrLn $ "Parameter estimate after " <> show nIter <> " steps:"
  printTensor p
  pure p

runExample :: IO (Tensor '[1,2])
runExample = do
  -- Generate data w/ ground truth params
  putStrLn "True parameters"
  trueParam <- fromList [3.5, (-4.4)]
  printTensor trueParam
  dat <- genData trueParam

  -- Setup GD
  p0 :: Tensor '[1, 2] <- fromList [0, 0]
  iters <- gradientDescent dat p0 0.0005 0.0001

  -- Results
  runN iters 10000

  -- -- peek at value w/o dispRaw pretty-printing
  -- putStrLn "Peek at raw pointer value of 2nd parameter:"
  -- let testVal = unsafePerformIO $ do
  --       withForeignPtr (tdsTensor res) (\pPtr -> pure $ c_THTensor_get2d pPtr 0 1)
  -- print testVal

main :: IO ()
main = do
  putStrLn "\nRun #1"
  putStrLn "\nRun #2 using the same random seed"
  _ <- runExample
  pure ()
