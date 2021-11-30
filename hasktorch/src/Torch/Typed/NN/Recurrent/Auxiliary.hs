{-# LANGUAGE DeriveGeneric #-}

module Torch.Typed.NN.Recurrent.Auxiliary where

import GHC.Generics
import Torch.Functional (mulScalar, subScalar)
import Torch.Tensor

data RNNInitialization
  = ConstantInitialization
  | LearnedInitialization
  deriving (Show, Generic)

-- TODO: This is taken from the initializers example code and should be replaced with cannonical,
-- tested versions. However, even a potentially incorrect implementation will likely perform
-- better than an ad-hoc random-normal distribution.

-- | Fan-in / Fan-out scaling calculation
calculateFan :: [Int] -> (Int, Int)
calculateFan shape
  | dimT < 2 =
    error
      "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
  | dimT == 2 =
    (numInputFmaps, numOutputFmaps)
  | otherwise =
    (numInputFmaps * receptiveFieldSize, numOutputFmaps * receptiveFieldSize)
  where
    dimT = length shape
    numInputFmaps = shape !! 1
    numOutputFmaps = shape !! 0
    receptiveFieldSize = product $ tail shape

-- | Xavier Initialization - Uniform
xavierUniformFIXME :: Tensor -> Float -> [Int] -> IO Tensor
xavierUniformFIXME init gain shape =
  pure $
    subScalar bound $
      mulScalar (bound * 2.0) init
  where
    (fanIn, fanOut) = calculateFan shape
    std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
    bound = sqrt 3.0 * std
