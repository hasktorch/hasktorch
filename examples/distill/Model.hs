{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Model where

import GHC.Generics

import Prelude hiding (exp, log)
import Torch

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP { 
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

mlpTemp :: Float -> MLP -> Tensor -> Tensor
mlpTemp temperature MLP{..} input = 
    logSoftmaxTemp (asTensor temperature)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input
  where
    logSoftmaxTemp t z = (z/t) - log (sumDim (Dim 1) KeepDim Float (exp (z/t)))

instance Parameterized MLP
instance HasForward MLP Tensor Tensor where
    forward = mlpTemp 1.0

instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)
