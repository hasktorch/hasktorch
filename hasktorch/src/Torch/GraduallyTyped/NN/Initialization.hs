{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}

module Torch.GraduallyTyped.NN.Initialization where

import Control.Monad.State.Strict (MonadState (state), runState)
import Torch.GraduallyTyped.DType (DataType (UncheckedDataType))
import Torch.GraduallyTyped.Device (Device (UncheckedDevice))
import Torch.GraduallyTyped.Layout (Layout (UncheckedLayout))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (Independent))
import Torch.GraduallyTyped.Shape (DimType (..), Shape (UncheckedShape))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Note: Identity = linear w/o activation
data NonLinearity = Identity | Sigmoid | Tanh | Relu | LeakyRelu Float

data FanMode = FanIn | FanOut

errorPrefix :: String
errorPrefix = "Error during tensor initialization. "

-- | Gain scaling value for He initialization
calculateGain :: NonLinearity -> Float
calculateGain Identity = 1.0
calculateGain Sigmoid = 1.0
calculateGain Tanh = 5.0 / 3
calculateGain Relu = sqrt 2.0
calculateGain (LeakyRelu param) = sqrt (2.0 / (1.0 + (param) ^^ 2))

dimSize :: DimType String Integer -> Integer
dimSize (Named _) = error $ errorPrefix <> "Cannot determine size of dimension."
dimSize (Sized size) = size
dimSize (NamedSized _ size) = size

-- | Fan-in / Fan-out scaling calculation
calculateFan ::
  [DimType String Integer] ->
  (Integer, Integer)
calculateFan shape =
  if dimT < 2
    then error $ errorPrefix <> "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    else
      if dimT == 2
        then
          ( numInputFmaps,
            numOutputFmaps
          )
        else
          ( numInputFmaps * receptiveFieldSize,
            numOutputFmaps * receptiveFieldSize
          )
  where
    dimT = length shape
    numInputFmaps = dimSize $ shape !! 1 
    numOutputFmaps = dimSize $ shape !! 0
    receptiveFieldSize = product $ dimSize <$> tail shape

-- | Xavier Initialization - Uniform
xavierUniform ::
  forall gain.
  (Num gain, Floating gain) =>
  gain ->
  [DimType String Integer] ->
  Generator 'UncheckedDevice ->
  ( Tensor 'Independent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape,
    Generator 'UncheckedDevice
  )
xavierUniform gain shape = runState $ do
  init <- _randIO' shape
  pure $ subScalar bound $ mulScalar (bound * 2.0) init
  where
    (fanIn, fanOut) = calculateFan shape
    std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
    bound = sqrt 3.0 * std

-- | Xavier Initialization - Normal
xavierNormal ::
  forall gain.
  (Num gain, Floating gain) =>
  gain ->
  [DimType String Integer] ->
  Generator 'UncheckedDevice ->
  ( Tensor 'Independent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape,
    Generator 'UncheckedDevice
  )
xavierNormal gain shape = runState $ do
  init <- _randnIO' shape
  pure $ mulScalar std init
  where
    (fanIn, fanOut) = calculateFan shape
    std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))

-- | Get fan in or fan out value depending on selected fan mode, used by Kaiming
getter :: forall a. FanMode -> ((a, a) -> a)
getter FanIn = fst
getter FanOut = snd

-- | Kaiming Initialization - Uniform
kaimingUniform ::
  FanMode ->
  NonLinearity ->
  [DimType String Integer] ->
  Generator 'UncheckedDevice ->
  ( Tensor 'Independent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape,
    Generator 'UncheckedDevice
  )
kaimingUniform mode nonlinearity shape = runState $ do
  init <- _randIO' shape
  pure $ subScalar bound $ mulScalar (bound * 2.0) init
  where
    gain = calculateGain nonlinearity
    fanValue = fromIntegral $ (getter mode) (calculateFan shape)
    std = gain / (sqrt fanValue)
    bound = (sqrt 3.0) * std

-- | Kaiming Initialization - Normal
kaimingNormal ::
  FanMode ->
  NonLinearity ->
  [DimType String Integer] ->
  Generator 'UncheckedDevice ->
  ( Tensor 'Independent 'UncheckedLayout 'UncheckedDevice 'UncheckedDataType 'UncheckedShape,
    Generator 'UncheckedDevice
  )
kaimingNormal mode nonlinearity shape = runState $ do
  init <- _randnIO' shape
  pure $ mulScalar std init
  where
    gain = calculateGain nonlinearity
    fanValue = fromIntegral $ (getter mode) (calculateFan shape)
    std = gain / (sqrt fanValue)
