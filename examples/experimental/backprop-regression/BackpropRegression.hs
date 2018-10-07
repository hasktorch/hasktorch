{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import System.IO (hFlush, stdout)
import Text.Printf

import Data.Generics.Product.Fields (field)
import GHC.Generics

import Numeric.Backprop as Bp
import Prelude as P
import Lens.Micro.TH
import Torch.Double as Torch hiding (add)
import Torch.Double.NN.Linear (Linear(..), linear)
import Torch.Double as Math hiding (Sum, add)
import qualified Torch.Core.Random as RNG

type NSamples = 2000

seedVal :: RNG.Seed
seedVal = 3141592653579

data Regression = Regression {
    linearLayer :: Linear 2 1
} deriving (Generic, Show)

instance Backprop Regression where
    add a b = Regression (Bp.add (linearLayer a) (linearLayer b))
    one _ = Regression (Bp.one undefined)
    zero _ = Regression (Bp.zero undefined)

newLayerWithBias :: All Dimensions '[d,d'] => Word -> IO (Tensor d, Tensor d')
newLayerWithBias n = do
  g <- newRNG
  let Just pair = ord2Tuple (-stdv, stdv)
  manualSeed g 10
  (,) <$> uniform g pair
      <*> uniform g pair
  where
    stdv :: Double
    stdv = 1 / P.sqrt (fromIntegral n)

newLinear :: forall o i . All KnownDim '[i,o] => IO (Linear i o)
newLinear = fmap Linear . newLayerWithBias $ dimVal (dim :: Dim i)

newRegression :: IO Regression
newRegression = Regression <$> newLinear

regression ::
    forall s . Reifies s W =>
    Double                  -- learning rate
    -> BVar s (Regression)  -- model architecture
    -> BVar s (Tensor '[2]) -- input
    -> BVar s (Tensor '[1]) -- output
regression learningRate modelArch input =
    linear learningRate (modelArch ^^. (field @"linearLayer")) input

genData :: Tensor '[1,2] -> IO (Tensor '[2, NSamples], Tensor '[NSamples])
genData param = do
  gen <- newRNG
  RNG.manualSeed gen seedVal
  let Just noiseScale = positive 2
      Just xScale = positive 10
  noise        :: Tensor '[NSamples] <- normal gen 0 noiseScale
  predictorVal :: Tensor '[NSamples] <- normal gen 0 xScale
  let x :: Tensor '[2, NSamples] = resizeAs (predictorVal `cat1d` (constant 1))
  let y :: Tensor '[NSamples]    = Math.cadd noise 1 (resizeAs (transpose2d (param !*! x)))
  pure (x, y)

-- TODO
train = undefined

epochs
  :: [(Tensor '[2], Tensor '[1])]
  -> HsReal
  -> Int
  -> [(Tensor '[2], Tensor '[1])]
  -> Regression
  -> IO ()
epochs test lr mx tset net0 = do
  printf "initial "
  testNet net0
  runEpoch 1 net0
  where
    runEpoch :: Int -> Regression -> IO ()
    runEpoch e net
      | e > mx    = pure ()
      | otherwise = do
        net' <- runBatch lr e 50 tset net
        testNet net'
        runEpoch (e + 1) net'
    testX = map fst test
    testY = map snd test
    testNet :: Regression -> IO ()
    testNet net = do
      printf ("[RMSE: %.1f%% / %d]") acc (length testY)
      hFlush stdout
     where
      preds = map (infer net) testX
      acc = undefined :: Double -- TODO

runBatch = undefined

infer :: Regression -> Tensor '[2] -> Tensor '[1]
infer architecture = evalBP2 (regression undefined) architecture

main :: IO ()
main = do
    let Just trueParam = fromList [3.5, -4.4]
    _ <- genData trueParam -- simulated data
    -- print simData
    _ <- newRegression -- instantiate network architecture
    putStrLn "Done"