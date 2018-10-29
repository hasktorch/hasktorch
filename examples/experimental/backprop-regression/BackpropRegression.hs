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

type BatchSize = 5

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
    BVar s (Regression)  -- model architecture
    -> BVar s (Tensor '[BatchSize, 2]) -- input
    -> BVar s (Tensor '[BatchSize, 1]) -- output
regression modelArch input =
    linearBatch 1 (modelArch ^^. (field @"linearLayer")) input

testSampler :: Generator -> IO (Tensor '[1])
testSampler gen = do
  let Just noiseScale = positive 1.0
  noise        :: Tensor '[1] <- normal gen 0 noiseScale
  let !foo = noise
  putStrLn ""
  print foo
  pure noise


genBatch' ::
  Generator -- RNG
  -> (Tensor '[2, 1], Double) -- (parameters, bias)
  -> IO (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])
genBatch' gen (param, bias) = do
  print gen
  let Just noiseScale = positive 1.0
      Just xScale = positive 10
  noise        :: Tensor '[BatchSize, 1] <- normal gen 0 noiseScale
  predictor1Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  predictor2Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  let biasTerm :: Tensor '[BatchSize, 1]  = (constant 1) ^* bias
  let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` predictor2Val)
  -- let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` (constant 1))
  let y :: Tensor '[BatchSize, 1] = (cadd noise 1 (resizeAs (transpose2d (x !*! param)))) + biasTerm
  print gen
  pure (x, y)

genBatch ::
  Generator -- RNG
  -> (Tensor '[2, 1], Double) -- (parameters, bias)
  -> IO (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1], Generator)
genBatch gen (param, bias) = do
  print gen
  let Just noiseScale = positive 1.0
      Just xScale = positive 10
  noise        :: Tensor '[BatchSize, 1] <- normal gen 0 noiseScale
  predictor1Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  predictor2Val :: Tensor '[BatchSize] <- normal gen 0 xScale
  let biasTerm :: Tensor '[BatchSize, 1]  = (constant 1) ^* bias
  let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` predictor2Val)
  -- let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictor1Val `cat1d` (constant 1))
  let y :: Tensor '[BatchSize, 1] = (cadd noise 1 (resizeAs (transpose2d (x !*! param)))) + biasTerm
  print gen
  pure (x, y, gen)

genBatches ::
  Generator -- RNG
  -> (Tensor '[2, 1], Double) -- (parameters, bias)
  -> Int -- number of batches
  -> IO [(Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])]
genBatches gen trueParam nBatch
  | nBatch == 0 = pure []
  | otherwise = do
      (newX, newY, newGen) <- genBatch gen trueParam
      next <- genBatches newGen trueParam (nBatch - 1)
      pure $ (newX, newY) : next

epochs ::
  HsReal                             -- learning rate
  -> Int                             -- max # of epochs
  -> [(Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])]    -- data to run batch on
  -> Regression                      -- initial model
  -> IO Regression
epochs lr maxEpochs tset net0 = do
  runEpoch 1 net0
  where
    runEpoch :: Int -> Regression -> IO Regression
    runEpoch epoch net
      | epoch > maxEpochs = pure net
      | otherwise = do
        (net', hist) <- foldM (trainStep lr) (net, []) tset
        let val =  P.sum $ catMaybes ((map ((`get1d` 0)) $ hist) :: [Maybe Double])
        printf ("[Epoch %d][mse %.4f]\n") epoch val
        hFlush stdout
        runEpoch (epoch + 1) net'

trainStep ::
  HsReal                                              -- learning rate
  -> (Regression, [(Tensor '[1])])                    -- (network, history)
  -> (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1]) -- input, output
  -> IO (Regression, [(Tensor '[1])])                 -- (updated network, history)
trainStep lr (net, hist) (x, y) = do
  pure (Bp.add net gnet, (out):hist)
  where
    gnet = Regression (ll ^* (-lr))
    (out, (Regression ll, _)) = backprop2 (mSECriterion y .: regression) net x

infer :: Regression -> Tensor '[BatchSize, 2] -> Tensor '[BatchSize, 1]
infer architecture = evalBP2 regression architecture

test :: IO ()
test = do
  Just (trueParam :: Tensor '[2, 1]) <- fromList [24.5, -80.4]
  let trueBias = 52.4
  gen <- newRNG
  -- batches <- mapM (\_ -> testSampler gen)  ([0..10] :: [Integer]) -- simulated data
  batches <- mapM (\_ -> genBatch' gen (trueParam, trueBias))  ([0..5] :: [Integer]) -- simulated data
  print $ head batches
  print $ last batches
  pure ()

main :: IO ()
main = do
    -- NOTE - estimate seems to get 1/2 of bias term because linear layer has a nuilt in bias term
    Just trueParam <- fromList [24.5, -80.4]
    let trueBias = 52.4
    gen <- newRNG
    RNG.manualSeed gen seedVal
    batches <- mapM (\_ -> genBatch' gen (trueParam, trueBias))  ([0..10] :: [Integer]) -- simulated data
    -- batches <- genBatches gen (trueParam, trueBias) 10

    putStrLn "\nFirst batch:"
    print $ head $ batches
    -- print $ head $ tail $ batches
    -- print $ head $ tail $ tail $ batches
    putStrLn "\nLast batch:"
    print $ last batches
    net0 <- Regression <$> newLinear -- instantiate network architecture
    let plr = 0.001
    let epos = 2
    net <- epochs plr epos batches net0
    let (estParam, estBias) = getTensors $ linearLayer net

    putStrLn "\n----\n"
    putStrLn "True Parameters:"
    print trueParam
    putStrLn "True Bias:"
    print trueBias
    print trueParam
    putStrLn "\n----\n"
    putStrLn "Parameter Estimates:"
    print estParam
    putStrLn "Bias Estimate:"
    print estBias


    gen2 <- newRNG
    bar <- mapM (\_ -> (testSampler gen2))  ([0..5] :: [Integer]) -- simulated data
    print bar

    putStrLn "\nDone"