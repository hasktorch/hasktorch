
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.TensorList where


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

import qualified Aten.Unmanaged.Type.TensorList as Unmanaged



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



