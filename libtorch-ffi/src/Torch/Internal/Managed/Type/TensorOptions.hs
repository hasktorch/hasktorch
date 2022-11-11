
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.TensorOptions where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.TensorOptions as Unmanaged





newTensorOptions_s
  :: ScalarType
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_s = _cast1 Unmanaged.newTensorOptions_s

tensorOptions_device_D
  :: ForeignPtr TensorOptions
  -> DeviceType
  -> IO (ForeignPtr TensorOptions)
tensorOptions_device_D = _cast2 Unmanaged.tensorOptions_device_D

tensorOptions_device_index_s
  :: ForeignPtr TensorOptions
  -> Int16
  -> IO (ForeignPtr TensorOptions)
tensorOptions_device_index_s = _cast2 Unmanaged.tensorOptions_device_index_s

tensorOptions_dtype_s
  :: ForeignPtr TensorOptions
  -> ScalarType
  -> IO (ForeignPtr TensorOptions)
tensorOptions_dtype_s = _cast2 Unmanaged.tensorOptions_dtype_s

tensorOptions_dtype
  :: ForeignPtr TensorOptions
  -> IO (ForeignPtr TensorOptions)
tensorOptions_dtype = _cast1 Unmanaged.tensorOptions_dtype

tensorOptions_layout_L
  :: ForeignPtr TensorOptions
  -> Layout
  -> IO (ForeignPtr TensorOptions)
tensorOptions_layout_L = _cast2 Unmanaged.tensorOptions_layout_L

tensorOptions_requires_grad_b
  :: ForeignPtr TensorOptions
  -> CBool
  -> IO (ForeignPtr TensorOptions)
tensorOptions_requires_grad_b = _cast2 Unmanaged.tensorOptions_requires_grad_b

tensorOptions_has_device
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_device = _cast1 Unmanaged.tensorOptions_has_device

tensorOptions_device_index
  :: ForeignPtr TensorOptions
  -> IO (Int32)
tensorOptions_device_index = _cast1 Unmanaged.tensorOptions_device_index

tensorOptions_has_dtype
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_dtype = _cast1 Unmanaged.tensorOptions_has_dtype

tensorOptions_layout
  :: ForeignPtr TensorOptions
  -> IO (Layout)
tensorOptions_layout = _cast1 Unmanaged.tensorOptions_layout

tensorOptions_has_layout
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_layout = _cast1 Unmanaged.tensorOptions_has_layout

tensorOptions_requires_grad
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_requires_grad = _cast1 Unmanaged.tensorOptions_requires_grad

tensorOptions_has_requires_grad
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_requires_grad = _cast1 Unmanaged.tensorOptions_has_requires_grad

tensorOptions_backend
  :: ForeignPtr TensorOptions
  -> IO (Backend)
tensorOptions_backend = _cast1 Unmanaged.tensorOptions_backend

dtype_s
  :: ScalarType
  -> IO (ForeignPtr TensorOptions)
dtype_s = _cast1 Unmanaged.dtype_s

layout_L
  :: Layout
  -> IO (ForeignPtr TensorOptions)
layout_L = _cast1 Unmanaged.layout_L

device_D
  :: DeviceType
  -> IO (ForeignPtr TensorOptions)
device_D = _cast1 Unmanaged.device_D

device_index_s
  :: Int16
  -> IO (ForeignPtr TensorOptions)
device_index_s = _cast1 Unmanaged.device_index_s

requires_grad_b
  :: CBool
  -> IO (ForeignPtr TensorOptions)
requires_grad_b = _cast1 Unmanaged.requires_grad_b

