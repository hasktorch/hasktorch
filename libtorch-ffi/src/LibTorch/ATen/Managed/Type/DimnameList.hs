
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.DimnameList where


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

import qualified LibTorch.ATen.Unmanaged.Type.DimnameList as Unmanaged



newDimnameList
  :: IO (ForeignPtr DimnameList)
newDimnameList = cast0 Unmanaged.newDimnameList





dimnameList_empty
  :: ForeignPtr DimnameList
  -> IO (CBool)
dimnameList_empty = cast1 Unmanaged.dimnameList_empty

dimnameList_size
  :: ForeignPtr DimnameList
  -> IO (CSize)
dimnameList_size = cast1 Unmanaged.dimnameList_size

dimnameList_at_s
  :: ForeignPtr DimnameList
  -> CSize
  -> IO (ForeignPtr Dimname)
dimnameList_at_s = cast2 Unmanaged.dimnameList_at_s

dimnameList_push_back_n
  :: ForeignPtr DimnameList
  -> ForeignPtr Dimname
  -> IO (())
dimnameList_push_back_n = cast2 Unmanaged.dimnameList_push_back_n



