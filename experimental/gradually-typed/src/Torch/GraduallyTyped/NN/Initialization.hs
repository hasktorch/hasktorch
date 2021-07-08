{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -Wno-typed-holes #-}
{-# OPTIONS_GHC -fdefer-typed-holes #-}

module Torch.GraduallyTyped.NN.Initialization where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Singletons (SingKind (fromSing))
import GHC.Generics (Generic)
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (SDevice)
import Torch.GraduallyTyped.Layout (SLayout)
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (SGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), dimSize)
import Torch.GraduallyTyped.Shape.Type (SShape)
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Note: Identity = linear w/o activation
data ForNonLinearity = ForIdentity | ForSigmoid | ForTanh | ForRelu | ForLeakyRelu Float
  deriving stock (Eq, Ord, Show, Generic)

data FanMode = FanIn | FanOut
  deriving stock (Eq, Ord, Show, Generic)

errorPrefix :: String
errorPrefix = "Error during tensor initialization. "

-- | Gain scaling value for He initialization
calculateGain :: ForNonLinearity -> Float
calculateGain ForIdentity = 1
calculateGain ForSigmoid = 1
calculateGain ForTanh = 5 / 3
calculateGain ForRelu = sqrt 2
calculateGain (ForLeakyRelu param) = sqrt (2 / (1 + param ^^ 2))

-- | Fan-in / Fan-out scaling calculation
calculateFan ::
  [Dim String Integer] ->
  (Integer, Integer)
calculateFan shape
  | dimT < 2 = error $ errorPrefix <> "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
  | dimT == 2 =
    ( numInputFmaps,
      numOutputFmaps
    )
  | otherwise =
    ( numInputFmaps * receptiveFieldSize,
      numOutputFmaps * receptiveFieldSize
    )
  where
    dimT = length shape
    numOutputFmaps : numInputFmaps : _ = dimSize <$> shape
    receptiveFieldSize = product $ dimSize <$> tail shape

-- | Xavier uniform initialization
sXavierUniform ::
  forall gradient layout device dataType shape gain device'.
  ( Num gain,
    Floating gain,
    Scalar gain
  ) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  gain ->
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
sXavierUniform reqGradient layout device dataType shape gain =
  let dims =
        fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
          . forgetIsChecked
          . fromSing
          $ shape
      (fanIn, fanOut) = calculateFan dims
      std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
      bound = sqrt 3 * std
   in runIxState $
        IxState (sRandn reqGradient layout device dataType shape)
          >>>= \init -> ireturn $ (init `mulScalar` (bound * 2)) `subScalar` bound

-- | Xavier normal initialization
sXavierNormal ::
  forall gradient layout device dataType shape gain device'.
  ( Num gain,
    Floating gain,
    Scalar gain
  ) =>
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  gain ->
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
sXavierNormal reqGradient layout device dataType shape gain =
  let dims =
        fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
          . forgetIsChecked
          . fromSing
          $ shape
      (fanIn, fanOut) = calculateFan dims
      std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
   in runIxState $
        IxState (sRandn reqGradient layout device dataType shape)
          >>>= \init -> ireturn $ init `mulScalar` std

-- | Get fan in or fan out value depending on selected fan mode, used by Kaiming
getter :: forall a. FanMode -> ((a, a) -> a)
getter FanIn = fst
getter FanOut = snd

-- | Kaiming uniform initialization
sKaimingUniform ::
  forall gradient layout device dataType shape device'.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  FanMode ->
  ForNonLinearity ->
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
sKaimingUniform reqGradient layout device dataType shape fanMode nonLinearity =
  let dims =
        fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
          . forgetIsChecked
          . fromSing
          $ shape
      gain = calculateGain nonLinearity
      fanValue = fromIntegral $ getter fanMode (calculateFan dims)
      std = gain / sqrt fanValue
      bound = sqrt 3 * std
   in runIxState $
        IxState (sRandn reqGradient layout device dataType shape)
          >>>= \init -> ireturn $ (init `mulScalar` (bound * 2)) `subScalar` bound

-- | Kaiming normal initialization
sKaimingNormal ::
  forall gradient layout device dataType shape device'.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  SShape shape ->
  FanMode ->
  ForNonLinearity ->
  Generator device' ->
  (Tensor gradient layout (device <+> device') dataType shape, Generator (device <+> device'))
sKaimingNormal reqGradient layout device dataType shape fanMode nonLinearity =
  let dims =
        fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
          . forgetIsChecked
          . fromSing
          $ shape
      gain = calculateGain nonLinearity
      fanValue = fromIntegral $ getter fanMode (calculateFan dims)
      std = gain / sqrt fanValue
   in runIxState $
        IxState (sRandn reqGradient layout device dataType shape)
          >>>= \init -> ireturn $ init `mulScalar` std