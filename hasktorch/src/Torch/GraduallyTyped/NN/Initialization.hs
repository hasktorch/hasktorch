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
import Torch.GraduallyTyped.DType (DataType (UncheckedDataType))
import Torch.GraduallyTyped.Device (Device (UncheckedDevice))
import Torch.GraduallyTyped.Layout (Layout (UncheckedLayout))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (Independent))
import Torch.GraduallyTyped.Shape (DimType (..), Shape (UncheckedShape), WidenShapeF)
import Torch.GraduallyTyped.Tensor.Creation (CreateC, CreateF, create, randn, unCreate)
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

-- | Xavier uniform initialization
xavierUniform ::
  forall requiresGradient layout device dataType shape gain.
  ( CreateC (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    CreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    Num gain,
    Floating gain
  ) =>
  CreateF (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
xavierUniform =
  create
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
          std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
          bound = sqrt 3.0 * std
       in runState $ do
            init <-
              state $
                unCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ subScalar bound $ mulScalar (bound * 2.0) init

-- | Xavier normal initialization
xavierNormal ::
  forall requiresGradient layout device dataType shape gain.
  ( CreateC (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    CreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    Num gain,
    Floating gain
  ) =>
  CreateF (gain -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
xavierNormal =
  create
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
          std = gain * sqrt (2.0 / (fromIntegral fanIn + fromIntegral fanOut))
       in runState $ do
            init <-
              state $
                unCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ mulScalar std init

-- | Get fan in or fan out value depending on selected fan mode, used by Kaiming
getter :: forall a. FanMode -> ((a, a) -> a)
getter FanIn = fst
getter FanOut = snd

-- | Kaiming uniform initialization
kaimingUniform ::
  forall requiresGradient layout device dataType shape.
  ( CreateC (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    CreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
  ) =>
  CreateF (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
kaimingUniform =
  create
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
          bound = (sqrt 3.0) * std
       in runState $ do
            init <-
              state $
                unCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ subScalar bound $ mulScalar (bound * 2.0) init

-- | Kaiming normal initialization
kaimingNormal ::
  forall requiresGradient layout device dataType shape.
  ( CreateC (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape,
    CreateC (Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
  ) =>
  CreateF (FanMode -> NonLinearity -> Generator device -> (Tensor requiresGradient layout device dataType shape, Generator device)) requiresGradient layout device dataType shape
kaimingNormal =
  create
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
                unCreate @_ @requiresGradient @layout @device @dataType @shape
                  (randn @requiresGradient @layout @device @dataType @shape)
                  requiresGradient
                  layoutType
                  deviceType
                  dType
                  shape
            pure $ mulScalar std init
