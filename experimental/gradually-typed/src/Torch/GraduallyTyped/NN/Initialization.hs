{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeOperators #-}

module Torch.GraduallyTyped.NN.Initialization where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import GHC.Generics (Generic)
import Torch.GraduallyTyped.Internal.TensorOptions (tensorDims)
import Torch.GraduallyTyped.Random (Generator, SGetGeneratorDevice)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), dimSize)
import Torch.GraduallyTyped.Tensor.Creation (sRandn)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar, subScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
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
calculateGain (ForLeakyRelu param) = sqrt (2 / (1 + param ^^ (2 :: Integer)))

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
  forall gradient layout device dataType shape gain generatorDevice m.
  ( Num gain,
    Floating gain,
    Scalar gain,
    MonadThrow m,
    SGetGeneratorDevice generatorDevice
  ) =>
  TensorSpec gradient layout device dataType shape ->
  gain ->
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sXavierUniform tensorSpec@TensorSpec {..} gain =
  let dims = tensorDims tsShape
      (fanIn, fanOut) = calculateFan dims
      std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
      bound = sqrt 3 * std
   in runIxStateT $
        IxStateT (sRandn tensorSpec)
          >>>= \initTensor -> ilift $ do
            x <- initTensor `mulScalar` (bound * 2)
            x `subScalar` bound

-- | Xavier normal initialization
sXavierNormal ::
  forall gradient layout device dataType shape gain generatorDevice m.
  ( Num gain,
    Floating gain,
    Scalar gain,
    MonadThrow m,
    SGetGeneratorDevice generatorDevice
  ) =>
  TensorSpec gradient layout device dataType shape ->
  gain ->
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sXavierNormal tensorSpec@TensorSpec {..} gain =
  let dims = tensorDims tsShape
      (fanIn, fanOut) = calculateFan dims
      std = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))
   in runIxStateT $
        IxStateT (sRandn tensorSpec)
          >>>= \initTensor -> ilift $ initTensor `mulScalar` std

-- | Get fan in or fan out value depending on selected fan mode, used by Kaiming
getter :: forall a. FanMode -> ((a, a) -> a)
getter FanIn = fst
getter FanOut = snd

-- | Kaiming uniform initialization
sKaimingUniform ::
  forall gradient layout device dataType shape generatorDevice m.
  ( MonadThrow m,
    SGetGeneratorDevice generatorDevice
  ) =>
  TensorSpec gradient layout device dataType shape ->
  FanMode ->
  ForNonLinearity ->
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sKaimingUniform tensorSpec@TensorSpec {..} fanMode nonLinearity =
  let dims = tensorDims tsShape
      gain = calculateGain nonLinearity
      fanValue = fromIntegral $ getter fanMode (calculateFan dims)
      std = gain / sqrt fanValue
      bound = sqrt 3 * std
   in runIxStateT $
        IxStateT (sRandn tensorSpec)
          >>>= \initTensor -> ilift $ do
            x <- initTensor `mulScalar` (bound * 2)
            x `subScalar` bound

-- | Kaiming normal initialization
sKaimingNormal ::
  forall gradient layout device dataType shape generatorDevice m.
  ( MonadThrow m,
    SGetGeneratorDevice generatorDevice
  ) =>
  TensorSpec gradient layout device dataType shape ->
  FanMode ->
  ForNonLinearity ->
  Generator generatorDevice ->
  m (Tensor gradient layout (device <+> generatorDevice) dataType shape, Generator (device <+> generatorDevice))
sKaimingNormal tensorSpec@TensorSpec {..} fanMode nonLinearity =
  let dims = tensorDims tsShape
      gain = calculateGain nonLinearity
      fanValue = fromIntegral $ getter fanMode (calculateFan dims)
      std = gain / sqrt fanValue
   in runIxStateT $
        IxStateT (sRandn tensorSpec)
          >>>= \initTensor -> ilift $ initTensor `mulScalar` std
