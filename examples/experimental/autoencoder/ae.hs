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

import GHC.Generics (Generic)

type DataDim = 256

data Autoencoder = Autoencoder {
    enc1 :: Linear DataDim 64
    , enc2 :: Linear 256 64
    , enc3 :: Linear 64 32
    , dec1 :: Linear 32 64
    , dec2 :: Linear 64 DataDim
} deriving (Generic, Show)

instance Backprop Autoencoder where
    add a b = Autoencoder 
        (Bp.add (enc1 a) (enc1 b))
        (Bp.add (enc2 a) (enc2 b))
        (Bp.add (enc3 a) (enc3 b))
        (Bp.add (dec1 a) (dec2 b))
        (Bp.add (dec1 a) (dec2 b))
    one _ = Autoencoder (Bp.one undefined)
    zero _ = Autoencoder (Bp.zero undefined)

seedVal = 31415926535

newLayerWithBias :: All Dimensions '[d,d'] => Word -> IO (Tensor d, Tensor d')
newLayerWithBias n = do
  g <- newRNG
  let Just pair = ord2Tuple (-stdv, stdv)
  manualSeed g seedVal
  (,) <$> uniform g pair
      <*> uniform g pair
  where
    stdv :: Double
    stdv = 1 / P.sqrt (fromIntegral n)

newLinear :: forall o i . All KnownDim '[i,o] => IO (Linear i o)
newLinear = fmap Linear . newLayerWithBias $ dimVal (dim :: Dim i)

forward :: forall s . Reifies s W =>
    BVar s Autoencoder -- model architecture
    -> BVar s (Tensor '[BatchSize, 2]) -- input
    -> BVar s (Tensor '[BatchSize, 1]) -- output
forward modelArch input =
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
  let y :: Tensor '[BatchSize, 1] = (cadd noise 1 (resizeAs (transpose2d (x !*! param)))) + biasTerm
  pure (x, y)

main = do
    putStrLn "Done"
