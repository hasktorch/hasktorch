
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
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.TensorOptions as Unmanaged



newTensorOptions_s
  :: ScalarType
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_s = cast1 Unmanaged.newTensorOptions_s





tensorOptions_device_D
  :: ForeignPtr TensorOptions
  -> DeviceType
  -> IO (ForeignPtr TensorOptions)
tensorOptions_device_D = cast2 Unmanaged.tensorOptions_device_D

tensorOptions_device_index_s
  :: ForeignPtr TensorOptions
  -> Int16
  -> IO (ForeignPtr TensorOptions)
tensorOptions_device_index_s = cast2 Unmanaged.tensorOptions_device_index_s

tensorOptions_dtype_s
  :: ForeignPtr TensorOptions
  -> ScalarType
  -> IO (ForeignPtr TensorOptions)
tensorOptions_dtype_s = cast2 Unmanaged.tensorOptions_dtype_s

tensorOptions_dtype
  :: ForeignPtr TensorOptions
  -> IO (ForeignPtr TensorOptions)
tensorOptions_dtype = cast1 Unmanaged.tensorOptions_dtype

tensorOptions_layout_L
  :: ForeignPtr TensorOptions
  -> Layout
  -> IO (ForeignPtr TensorOptions)
tensorOptions_layout_L = cast2 Unmanaged.tensorOptions_layout_L

tensorOptions_requires_grad_b
  :: ForeignPtr TensorOptions
  -> CBool
  -> IO (ForeignPtr TensorOptions)
tensorOptions_requires_grad_b = cast2 Unmanaged.tensorOptions_requires_grad_b

tensorOptions_is_variable_b
  :: ForeignPtr TensorOptions
  -> CBool
  -> IO (ForeignPtr TensorOptions)
tensorOptions_is_variable_b = cast2 Unmanaged.tensorOptions_is_variable_b

tensorOptions_has_device
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_device = cast1 Unmanaged.tensorOptions_has_device

tensorOptions_device_index
  :: ForeignPtr TensorOptions
  -> IO (Int32)
tensorOptions_device_index = cast1 Unmanaged.tensorOptions_device_index

tensorOptions_has_dtype
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_dtype = cast1 Unmanaged.tensorOptions_has_dtype

tensorOptions_layout
  :: ForeignPtr TensorOptions
  -> IO (Layout)
tensorOptions_layout = cast1 Unmanaged.tensorOptions_layout

tensorOptions_has_layout
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_layout = cast1 Unmanaged.tensorOptions_has_layout

tensorOptions_requires_grad
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_requires_grad = cast1 Unmanaged.tensorOptions_requires_grad

tensorOptions_has_requires_grad
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_requires_grad = cast1 Unmanaged.tensorOptions_has_requires_grad

tensorOptions_is_variable
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_is_variable = cast1 Unmanaged.tensorOptions_is_variable

tensorOptions_has_is_variable
  :: ForeignPtr TensorOptions
  -> IO (CBool)
tensorOptions_has_is_variable = cast1 Unmanaged.tensorOptions_has_is_variable

tensorOptions_backend
  :: ForeignPtr TensorOptions
  -> IO (Backend)
tensorOptions_backend = cast1 Unmanaged.tensorOptions_backend



dtype_s
  :: ScalarType
  -> IO (ForeignPtr TensorOptions)
dtype_s = cast1 Unmanaged.dtype_s

layout_L
  :: Layout
  -> IO (ForeignPtr TensorOptions)
layout_L = cast1 Unmanaged.layout_L

device_D
  :: DeviceType
  -> IO (ForeignPtr TensorOptions)
device_D = cast1 Unmanaged.device_D

device_index_s
  :: Int16
  -> IO (ForeignPtr TensorOptions)
device_index_s = cast1 Unmanaged.device_index_s

requires_grad_b
  :: CBool
  -> IO (ForeignPtr TensorOptions)
requires_grad_b = cast1 Unmanaged.requires_grad_b

