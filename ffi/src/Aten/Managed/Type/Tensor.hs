
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.Tensor where


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

import qualified Aten.Unmanaged.Type.Tensor as Unmanaged



newTensor
  :: IO (ForeignPtr Tensor)
newTensor = cast0 Unmanaged.newTensor

newTensor_Tensor
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
newTensor_Tensor = cast1 Unmanaged.newTensor_Tensor





tensor_dim
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_dim = cast1 Unmanaged.tensor_dim

tensor_storage_offset
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_storage_offset = cast1 Unmanaged.tensor_storage_offset

tensor_defined
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_defined = cast1 Unmanaged.tensor_defined

tensor_reset
  :: ForeignPtr Tensor
  -> IO (())
tensor_reset = cast1 Unmanaged.tensor_reset

tensor_use_count
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_use_count = cast1 Unmanaged.tensor_use_count

tensor_weak_use_count
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_weak_use_count = cast1 Unmanaged.tensor_weak_use_count

tensor_ndimension
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_ndimension = cast1 Unmanaged.tensor_ndimension

tensor_is_contiguous
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_contiguous = cast1 Unmanaged.tensor_is_contiguous

tensor_nbytes
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_nbytes = cast1 Unmanaged.tensor_nbytes

tensor_itemsize
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_itemsize = cast1 Unmanaged.tensor_itemsize

tensor_element_size
  :: ForeignPtr Tensor
  -> IO (CSize)
tensor_element_size = cast1 Unmanaged.tensor_element_size

tensor_is_variable
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_variable = cast1 Unmanaged.tensor_is_variable

tensor_get_device
  :: ForeignPtr Tensor
  -> IO (Int64)
tensor_get_device = cast1 Unmanaged.tensor_get_device

tensor_is_cuda
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_cuda = cast1 Unmanaged.tensor_is_cuda

tensor_is_hip
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_hip = cast1 Unmanaged.tensor_is_hip

tensor_is_sparse
  :: ForeignPtr Tensor
  -> IO (CBool)
tensor_is_sparse = cast1 Unmanaged.tensor_is_sparse

tensor_print
  :: ForeignPtr Tensor
  -> IO (())
tensor_print = cast1 Unmanaged.tensor_print

