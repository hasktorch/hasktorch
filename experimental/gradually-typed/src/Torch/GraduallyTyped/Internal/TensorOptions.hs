{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.GraduallyTyped.Internal.TensorOptions where

import Data.Singletons (SingKind (..))
import Foreign.ForeignPtr (ForeignPtr)
import System.IO.Unsafe (unsafePerformIO)
import Torch.GraduallyTyped.DType (SDataType)
import Torch.GraduallyTyped.Device (DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (SLayout)
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), SShape)
import Torch.Internal.Cast (cast1, cast2)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen (kCPU)
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Type as ATen

newtype TensorOptions = TensorOptions (ForeignPtr ATen.TensorOptions)

instance Castable TensorOptions (ForeignPtr ATen.TensorOptions) where
  cast (TensorOptions opts) f = f opts
  uncast opts f = f $ TensorOptions opts

tensorOptions ::
  forall gradient layout device dataType.
  SGradient gradient ->
  SLayout layout ->
  SDevice device ->
  SDataType dataType ->
  TensorOptions
tensorOptions gradient layout device dataType = unsafePerformIO $ do
  opts :: TensorOptions <- cast1 ATen.newTensorOptions_s dType
  opts :: TensorOptions <- let b = requiresGradient == WithGradient in cast2 ATen.tensorOptions_requires_grad_b opts b
  opts :: TensorOptions <- withDevice deviceType opts
  opts :: TensorOptions <- cast2 ATen.tensorOptions_layout_L opts layoutType
  return opts
  where
    requiresGradient = forgetIsChecked . fromSing $ gradient
    layoutType = forgetIsChecked . fromSing $ layout
    deviceType = forgetIsChecked . fromSing $ device
    dType = forgetIsChecked . fromSing $ dataType

    withDevice CPU opts = cast2 ATen.tensorOptions_device_D opts ATen.kCPU
    withDevice (CUDA deviceIndex) opts = cast2 ATen.tensorOptions_device_index_s opts deviceIndex

tensorDims ::
  forall shape.
  SShape shape ->
  [Dim String Integer]
tensorDims =
  fmap (\(Dim name size) -> Dim (forgetIsChecked name) (forgetIsChecked size))
    . forgetIsChecked
    . fromSing
