{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import System.IO (hFlush, stdout)
import Text.Printf

import Control.Monad (foldM)
import Data.Maybe (catMaybes)
import Data.Generics.Product.Fields (field)
import GHC.Generics

import Numeric.Backprop as Bp
import Prelude as P
import Torch.Double as Torch hiding (add)
import Torch.Double.NN.Linear (Linear(..), linearBatch)
import qualified Torch.Core.Random as RNG

type BatchSize = 10

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

regression :: forall s . Reifies s W =>
    BVar s Regression  -- model architecture
    -> BVar s (Tensor '[BatchSize, 2]) -- input
    -> BVar s (Tensor '[BatchSize, 1]) -- output
regression modelArch input =
    linearBatch (modelArch ^^. (field @"linearLayer")) input

genBatch ::
  Generator -- RNG
  -> (Tensor '[2, 1], Double) -- (parameters, bias)
  -> IO (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])
genBatch gen (param, bias) = do
  let Just noiseScale = positive 1.0
      Just xScale = positive 10
  noise        :: Tensor '[BatchSize, 1] <- normal gen 0 noiseScale
  predictor1Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  predictor2Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  let biasTerm :: Tensor '[BatchSize, 1]  = (constant 1) ^* bias
  let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` predictor2Val)
  -- let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` (constant 1))
  let y :: Tensor '[BatchSize, 1] = (cadd noise 1 (resizeAs (transpose2d (x !*! param)))) + biasTerm
  pure (x, y)

trainStep ::
  HsReal                                              -- learning rate
  -> (Regression, [(Tensor '[1])])                    -- (network, history)
  -> (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1]) -- input, output
  -> IO (Regression, [(Tensor '[1])])                 -- (updated network, history)
trainStep learningRate (net, hist) (x, y) = do
  pure (Bp.add net gnet, (out):hist)
  where
    gnet = Regression (ll ^* (-learningRate))
    (out, (Regression ll, _)) = backprop2 (mSECriterion y .: regression) net x

epochs ::
  HsReal                                                -- learning rate
  -> Int                                                -- max # of epochs
  -> [(Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])] -- data to run batch on
  -> Regression                                         -- initial model
  -> IO Regression
epochs learningRate maxEpochs tset net0 = do
  runEpoch 0 net0
  where
    runEpoch :: Int -> Regression -> IO Regression
    runEpoch epoch net
      | epoch > maxEpochs = pure net
      | otherwise = do
        (net', hist) <- foldM (trainStep learningRate) (net, []) tset
        let val =  P.sum $ catMaybes ((map ((`get1d` 0)) $ hist) :: [Maybe Double])
        if epoch `mod` 50 == 0 then
          printf ("[Epoch %d][Loss %.4f]\n") epoch val
        else
          pure ()
        hFlush stdout
        runEpoch (epoch + 1) net'

infer :: Regression -> Tensor '[BatchSize, 2] -> Tensor '[BatchSize, 1]
infer architecture = evalBP2 regression architecture

main :: IO ()
main = do

    -- model parameters
    Just trueParam <- fromList [24.5, -80.4]
    let trueBias = 52.4
    let numBatch = 2
    let learningRate = 0.005
    let numEpochs = 200

    -- produce simulated data
    gen <- newRNG
    RNG.manualSeed gen seedVal
    batches <- mapM (\_ -> genBatch gen (trueParam, trueBias))  ([1..numBatch] :: [Integer])
    -- FIXME: fix RNG advancement bug for cases where numBatch > 1

    -- train model
    net0 <- Regression <$> newLinear -- instantiate network architecture
    putStrLn "\nTraining ========================================\n"
    net <- epochs learningRate numEpochs batches net0
    let (estParam, estBias) = getTensors $ linearLayer net

    -- pshow results
    putStrLn "\nModel Parameters ================================"
    putStrLn "\nTrue Parameters:\n"
    print trueParam
    putStrLn "\nTrue Bias:\n"
    print trueBias
    putStrLn "\nEstimated Parameters ============================"
    putStrLn "\nParameter Estimates:\n"
    print estParam
    putStrLn "\nBias Estimate:\n"
    print estBias
