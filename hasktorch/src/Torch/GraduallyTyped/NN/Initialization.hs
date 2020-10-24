{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-typed-holes #-}
{-# OPTIONS_GHC -fdefer-typed-holes #-}

module Torch.GraduallyTyped.NN.Initialization where

import Control.Monad.State.Strict (MonadState (state), runState)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (DimType (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), randn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Note: Identity = linear w/o activation
data NonLinearity = Identity | Sigmoid | Tanh | Relu | LeakyRelu Float

data FanMode = FanIn | FanOut

errorPrefix :: String
errorPrefix = "Error during tensor initialization. "

-- | Gain scaling value for He initialization
calculateGain :: NonLinearity -> Float
calculateGain Identity = 1
calculateGain Sigmoid = 1
calculateGain Tanh = 5 / 3
calculateGain Relu = sqrt 2
calculateGain (LeakyRelu param) = sqrt (2 / (1 + param ^^ 2))

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

-- | Xavier uniform initialization
xavierUniform ::
  forall requiresGradient layout device dataType shape gain.
  ( WithCreateC (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    WithCreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    Num gain,
    Floating gain,
    Scalar gain
  ) =>
  WithCreateF (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
xavierUniform =
  withCreate
    @(gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device))
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType shape gain =
      let (fanIn, fanOut) = calculateFan shape
          std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
          bound = sqrt 3 * std
       in runState $ do
            init <-
              state $
                withoutCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ (init `mulScalar` (bound * 2)) `subScalar` bound

-- | Xavier normal initialization
xavierNormal ::
  forall requiresGradient layout device dataType shape gain.
  ( WithCreateC (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    WithCreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    Num gain,
    Floating gain,
    Scalar gain
  ) =>
  WithCreateF (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
xavierNormal =
  withCreate
    @(gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device))
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType shape gain =
      let (fanIn, fanOut) = calculateFan shape
          std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
       in runState $ do
            init <-
              state $
                withoutCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ init `mulScalar` std

-- | Get fan in or fan out value depending on selected fan mode, used by Kaiming
getter :: forall a. FanMode -> ((a, a) -> a)
getter FanIn = fst
getter FanOut = snd

-- | Kaiming uniform initialization
kaimingUniform ::
  forall requiresGradient layout device dataType shape.
  ( WithCreateC (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    WithCreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
  ) =>
  WithCreateF (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
kaimingUniform =
  withCreate
    @(FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device))
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType shape fanMode nonLinearity =
      let gain = calculateGain nonLinearity
          fanValue = fromIntegral $ (getter fanMode) (calculateFan shape)
          std = gain / (sqrt fanValue)
          bound = (sqrt 3) * std
       in runState $ do
            init <-
              state $
                withoutCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ (init `mulScalar` (bound * 2)) `subScalar` bound

-- | Kaiming normal initialization
kaimingNormal ::
  forall requiresGradient layout device dataType shape.
  ( WithCreateC (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    WithCreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
  ) =>
  WithCreateF (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
kaimingNormal =
  withCreate
    @(FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device))
    @requiresGradient
    @layout
    @device
    @dataType
    @shape
    go
  where
    go requiresGradient layoutType deviceType dType shape fanMode nonLinearity =
      let gain = calculateGain nonLinearity
          fanValue = fromIntegral $ (getter fanMode) (calculateFan shape)
          std = gain / (sqrt fanValue)
       in runState $ do
            init <-
              state $
                withoutCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ init `mulScalar` std
