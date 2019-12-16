
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.DimnameList where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple
import ATen.Unmanaged.Type.StdString
import ATen.Unmanaged.Type.Dimname
import ATen.Unmanaged.Type.DimnameList

import qualified ATen.Unmanaged.Type.DimnameList as Unmanaged



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



