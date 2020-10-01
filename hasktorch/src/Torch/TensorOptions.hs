{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.TensorOptions where

import Data.Int
import Foreign.ForeignPtr
import System.IO.Unsafe
import Torch.DType
import Torch.Device
import Torch.Internal.Cast
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen
import qualified Torch.Internal.Managed.Type.Context as ATen
import qualified Torch.Internal.Managed.Type.TensorOptions as ATen
import qualified Torch.Internal.Type as ATen
import Torch.Layout

type ATenTensorOptions = ForeignPtr ATen.TensorOptions

newtype TensorOptions = TensorOptions ATenTensorOptions deriving (Show)

instance Castable TensorOptions ATenTensorOptions where
  cast (TensorOptions aten_opts) f = f aten_opts
  uncast aten_opts f = f $ TensorOptions aten_opts

defaultOpts :: TensorOptions
defaultOpts =
  TensorOptions $ unsafePerformIO $ ATen.newTensorOptions_s ATen.kFloat

withDType :: DType -> TensorOptions -> TensorOptions
withDType dtype opts =
  unsafePerformIO $ cast2 ATen.tensorOptions_dtype_s opts dtype

withDevice :: Device -> TensorOptions -> TensorOptions
withDevice Device {..} opts = unsafePerformIO $ do
  hasCUDA <- cast0 ATen.hasCUDA
  withDevice' deviceType deviceIndex hasCUDA opts
  where
    withDeviceType :: DeviceType -> TensorOptions -> IO TensorOptions
    withDeviceType dt opts = cast2 ATen.tensorOptions_device_D opts dt
    withDeviceIndex :: Int16 -> TensorOptions -> IO TensorOptions
    withDeviceIndex di opts = cast2 ATen.tensorOptions_device_index_s opts di -- careful, this somehow implies deviceType = CUDA
    withDevice' ::
      DeviceType -> Int16 -> Bool -> TensorOptions -> IO TensorOptions
    withDevice' CPU 0 False opts = pure opts
    withDevice' CPU 0 True opts = pure opts >>= withDeviceType CPU
    withDevice' CUDA di True opts | di >= 0 = pure opts >>= withDeviceIndex di
    withDevice' dt di _ _ =
      error $ "cannot move tensor to \"" <> show dt <> ":" <> show di <> "\""

withLayout :: Layout -> TensorOptions -> TensorOptions
withLayout layout opts =
  unsafePerformIO $ cast2 ATen.tensorOptions_layout_L opts layout
