
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.IntArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import LibTorch.ATen.Type
import LibTorch.ATen.Class
import LibTorch.ATen.Cast
import LibTorch.ATen.Unmanaged.Type.Generator
import LibTorch.ATen.Unmanaged.Type.IntArray
import LibTorch.ATen.Unmanaged.Type.Scalar
import LibTorch.ATen.Unmanaged.Type.Storage
import LibTorch.ATen.Unmanaged.Type.Tensor
import LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Unmanaged.Type.TensorOptions
import LibTorch.ATen.Unmanaged.Type.Tuple
import LibTorch.ATen.Unmanaged.Type.StdString
import LibTorch.ATen.Unmanaged.Type.Dimname
import LibTorch.ATen.Unmanaged.Type.DimnameList

import qualified LibTorch.ATen.Unmanaged.Type.IntArray as Unmanaged



newIntArray
  :: IO (ForeignPtr IntArray)
newIntArray = cast0 Unmanaged.newIntArray





intArray_empty
  :: ForeignPtr IntArray
  -> IO (CBool)
intArray_empty = cast1 Unmanaged.intArray_empty

intArray_size
  :: ForeignPtr IntArray
  -> IO (CSize)
intArray_size = cast1 Unmanaged.intArray_size

intArray_at_s
  :: ForeignPtr IntArray
  -> CSize
  -> IO (Int64)
intArray_at_s = cast2 Unmanaged.intArray_at_s

intArray_push_back_l
  :: ForeignPtr IntArray
  -> Int64
  -> IO (())
intArray_push_back_l = cast2 Unmanaged.intArray_push_back_l



