{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.GraduallyTyped.Internal.TensorOptions where

import Data.Int (Int16)
import Foreign.ForeignPtr (ForeignPtr)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType ()
import Torch.GraduallyTyped.Device (DeviceType (..))
import Torch.GraduallyTyped.Layout (LayoutType)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
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
  RequiresGradient ->
  LayoutType ->
  DeviceType Int16 ->
  DType ->
  TensorOptions
tensorOptions requiresGradient layoutType deviceType dType = unsafePerformIO $ do
  opts :: TensorOptions <- cast1 ATen.newTensorOptions_s dType
  opts :: TensorOptions <- let b = requiresGradient == Independent in cast2 ATen.tensorOptions_requires_grad_b opts b
  opts :: TensorOptions <- withDevice deviceType opts
  opts :: TensorOptions <- cast2 ATen.tensorOptions_layout_L opts layoutType
  return opts
  where
    withDevice CPU opts = cast2 ATen.tensorOptions_device_D opts ATen.kCPU
    withDevice (CUDA deviceIndex) opts = cast2 ATen.tensorOptions_device_index_s opts deviceIndex
