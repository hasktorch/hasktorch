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
import Torch.Double as Math hiding (add)
import qualified Torch.Core.Random as RNG

type BatchSize = 40

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

genData :: Generator -> Tensor '[2, 1] -> IO (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])
genData gen param = do
  RNG.manualSeed gen seedVal
  let Just noiseScale = positive 0.1
      Just xScale = positive 10
  noise        :: Tensor '[BatchSize, 1] <- normal gen 0 noiseScale
  predictorVal :: Tensor '[BatchSize] <- normal gen 0 xScale
  let x :: Tensor '[BatchSize, 2] = transpose2d $ resizeAs (predictorVal `cat1d` (constant 1))
  let y :: Tensor '[BatchSize, 1]    = Math.cadd noise 1 (resizeAs (transpose2d (x !*! param)))
  pure (x, y)

epochs
  :: 
  HsReal                          -- learning rate
  -> Int                             -- max # of epochs
  -> [(Tensor '[BatchSize, 2], Tensor '[BatchSize, 1])]    -- data to run batch on
  -> Regression                      -- initial model
  -> IO ()
epochs lr mx tset net0 = do
  runEpoch 1 net0
  where
    runEpoch :: Int -> Regression -> IO ()
    runEpoch e net
      | e > mx    = pure ()
      | otherwise = do
        (net', hist) <- foldM (trainStep lr) (net, []) tset
        let val =  P.sum $ catMaybes ((map ((`get1d` 0)) $ hist) :: [Maybe Double])
        printf ("[Epoch %d][mse %.4f]\n") e val
        hFlush stdout
        runEpoch (e + 1) net'

trainStep :: 
  HsReal                                              -- learning rate
  -> (Regression, [(Tensor '[1])])                    -- (network, history)
  -> (Tensor '[BatchSize, 2], Tensor '[BatchSize, 1]) -- input, output
  -> IO (Regression, [(Tensor '[1])])                 -- (updated network, history)
trainStep lr (net, hist) (x, y) = do
  pure (Bp.add net gnet, (out):hist)
  where
    gnet = Regression (ll ^* (-lr))
    (out, (Regression ll, _))
      = backprop2
        (mSECriterion y .: regression) 
        net x

infer :: Regression -> Tensor '[BatchSize, 2] -> Tensor '[BatchSize, 1]
infer architecture = evalBP2 regression architecture

main :: IO ()
main = do
    Just trueParam <- fromList [3.5, -4.4]
    gen <- newRNG
    batches <- mapM (\_ -> genData gen trueParam) ([0..100] :: [Integer]) -- simulated data
    net0 <- Regression <$> newLinear -- instantiate network architecture
    let plr = 0.0001
    let epos = 200
    _ <- epochs plr epos batches net0

    putStrLn "Done"

-- | set rewind marker for 'clearLn'
setRewind :: String
setRewind = "\r"
