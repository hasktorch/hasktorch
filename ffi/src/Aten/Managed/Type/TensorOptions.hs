
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.TensorOptions where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import Aten.Unmanaged.Type.Generator
import Aten.Unmanaged.Type.IntArray
import Aten.Unmanaged.Type.Scalar
import Aten.Unmanaged.Type.SparseTensorRef
import Aten.Unmanaged.Type.Storage
import Aten.Unmanaged.Type.Tensor
import Aten.Unmanaged.Type.TensorList
import Aten.Unmanaged.Type.TensorOptions
import Aten.Unmanaged.Type.Tuple

import qualified Aten.Unmanaged.Type.TensorOptions as Unmanaged



newTensorOptions_D
  :: DeviceType
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_D = cast1 Unmanaged.newTensorOptions_D

newTensorOptions_B
  :: Backend
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_B = cast1 Unmanaged.newTensorOptions_B

newTensorOptions_s
  :: ScalarType
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_s = cast1 Unmanaged.newTensorOptions_s

newTensorOptions_L
  :: Layout
  -> IO (ForeignPtr TensorOptions)
newTensorOptions_L = cast1 Unmanaged.newTensorOptions_L





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

