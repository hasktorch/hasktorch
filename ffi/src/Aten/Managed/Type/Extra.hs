
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.Extra where


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

import qualified Aten.Unmanaged.Type.Extra as Unmanaged

tensor_assign1_l
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO ()
tensor_assign1_l  = cast3 Unmanaged.tensor_assign1_l

tensor_assign2_l
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO ()
tensor_assign2_l = cast4 Unmanaged.tensor_assign2_l

tensor_assign1_d
  :: ForeignPtr Tensor
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign1_d = cast3 Unmanaged.tensor_assign1_d

tensor_assign2_d
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> CDouble
  -> IO ()
tensor_assign2_d = cast4 Unmanaged.tensor_assign2_d

tensor_assign1_t
  :: ForeignPtr Tensor
  -> Int64
  -> ForeignPtr Tensor
  -> IO ()
tensor_assign1_t  = cast3 Unmanaged.tensor_assign1_t

tensor_assign2_t
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> ForeignPtr Tensor
  -> IO ()
tensor_assign2_t = cast4 Unmanaged.tensor_assign2_t

