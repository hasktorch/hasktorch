
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.TensorList where


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

import qualified LibTorch.ATen.Unmanaged.Type.TensorList as Unmanaged



newTensorList
  :: IO (ForeignPtr TensorList)
newTensorList = cast0 Unmanaged.newTensorList





tensorList_empty
  :: ForeignPtr TensorList
  -> IO (CBool)
tensorList_empty = cast1 Unmanaged.tensorList_empty

tensorList_size
  :: ForeignPtr TensorList
  -> IO (CSize)
tensorList_size = cast1 Unmanaged.tensorList_size

tensorList_at_s
  :: ForeignPtr TensorList
  -> CSize
  -> IO (ForeignPtr Tensor)
tensorList_at_s = cast2 Unmanaged.tensorList_at_s

tensorList_push_back_t
  :: ForeignPtr TensorList
  -> ForeignPtr Tensor
  -> IO (())
tensorList_push_back_t = cast2 Unmanaged.tensorList_push_back_t



