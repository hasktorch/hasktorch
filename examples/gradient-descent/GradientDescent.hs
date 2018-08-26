-------------------------------------------------------------------------------
-- Toy gradient descent example
--
-- This example illustrates using basic linear algebra functions in a toy test
-- example linear regression optimization. It's mainly intended for 
-- familiarity with basic matrix/vector operations and is not optimized.
--
-- For something more complex than this toy case, a backprop-based 
-- implementation would probably be preferable. 
-- 
-------------------------------------------------------------------------------

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Data.Proxy
import GHC.TypeLits
import Numeric.Dimensions hiding (N)
import Data.Monoid ((<>))
import Control.Monad
import Lens.Micro

import Torch.Double hiding (N)
import qualified Torch.Double as Math
import qualified Torch.Core.Random as RNG

-- N, sample size, and P dimension of model parameters
type N = 2000
type P = '[1, 2]
type Precision = Double

seedVal :: RNG.Seed
seedVal = 3141592653579

-- | Generate simulated data by:
-- - Sampling random predictor values from ~ N(0, 10)
-- - The `param` function parameter specifies ground truth values 
--   applied to the output function `y = (b_1 x + b_0)`
-- - For vectorization, b_0 is represented as a coefficient
--   on a second predictor dimension that is always 1
-- - The x vector is thus represented as a 2xN matrix
--   [[x_0 1], [x_1 1], ...]
-- - Add a noise component to the observed y values from ~ N(0, 2)
genData :: Tensor '[1,2] -> IO (Tensor '[2, N], Tensor '[N])
genData param = do
  gen <- newRNG
  RNG.manualSeed gen seedVal
  let Just noiseScale = positive 2
      Just xScale = positive 10
  noise        :: Tensor '[N] <- normal gen 0 noiseScale
  predictorVal :: Tensor '[N] <- normal gen 0 xScale
  let x :: Tensor '[2, N] = resizeAs (predictorVal `cat1d` (constant 1))
  let y :: Tensor '[N]    = Math.cadd noise 1 (resizeAs (transpose2d (param !*! x)))

  pure (x, y)

-- | Loss is defined as the sum of squared errors
loss :: (Tensor '[2,N], Tensor '[N]) -> Tensor '[1, 2] -> IO Precision
loss (x, y) param = do
  let errors = y - resizeAs (param !*! x)
  (realToFrac . Math.sumall) <$> Math.square errors


-- | Gradient is 2/N (error) x
--   2 is from the derivative of the squared loss
--   N is from a mean-squared-error interpretation of the loss
--   normalizes the scale of the rate by the size of the dataset
--   err is from the loss definition, the multiplication by x'
--   is from the application of the chain rule
gradient
  :: forall n . (KnownDim n)
  => (Tensor '[2, n], Tensor '[n]) -> Tensor '[1, 2] -> IO (Tensor '[1, 2])
gradient (x, y) param = do
  let y' :: Tensor '[1, n] = resizeAs y
  let x' :: Tensor '[n, 2] = transpose2d x
  let m  :: Tensor '[1, 2] = resizeAs (err y' !*! x')
  pure $ (-2 / nsamp) *^ m
  where
    err :: Tensor '[1, n] -> Tensor '[1, n]
    err y' = y' - (param !*! x)
    nsamp :: Precision
    nsamp = realToFrac (dimVal (dim :: Dim n))

-- | Runs gradient descent until a specified threshold change
--   Note that laziness of evaluation is exploited so this sequence
--   does not run to completion of the iterations aren't evaluated
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


-- |  Run N iterations of gradient descent, `take` is used + printing
--    is used to force evaluation
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

-- | Define ground-truth parameters, generate data, then run GD procedure.
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

main :: IO ()
main = do
  putStrLn "\nRun #1"
  void runExample
