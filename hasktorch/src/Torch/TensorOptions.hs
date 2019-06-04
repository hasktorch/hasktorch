{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.TensorOptions where

import Foreign.ForeignPtr
import System.IO.Unsafe

import ATen.Cast
import ATen.Class (Castable(..))
import qualified ATen.Type as ATen
import qualified ATen.Const as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen

import Torch.DType

type ATenTensorOptions = ForeignPtr ATen.TensorOptions

data TensorOptions = TensorOptions ATenTensorOptions

instance Castable TensorOptions ATenTensorOptions where
  cast (TensorOptions aten_opts) f = f aten_opts
  uncast aten_opts f = f $ TensorOptions aten_opts

defaultOpts :: TensorOptions
defaultOpts = TensorOptions $ unsafePerformIO $ ATen.newTensorOptions_s ATen.kFloat

withDType :: DType -> TensorOptions -> TensorOptions
withDType dtype opts = unsafePerformIO $ (cast2 ATen.tensorOptions_dtype_s) opts dtype
