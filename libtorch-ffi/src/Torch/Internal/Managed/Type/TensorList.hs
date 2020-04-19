{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.TensorList where

import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import qualified Torch.Internal.Unmanaged.Type.TensorList as Unmanaged
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

newTensorList ::
  IO (ForeignPtr TensorList)
newTensorList = cast0 Unmanaged.newTensorList

tensorList_empty ::
  ForeignPtr TensorList ->
  IO (CBool)
tensorList_empty = cast1 Unmanaged.tensorList_empty

tensorList_size ::
  ForeignPtr TensorList ->
  IO (CSize)
tensorList_size = cast1 Unmanaged.tensorList_size

tensorList_at_s ::
  ForeignPtr TensorList ->
  CSize ->
  IO (ForeignPtr Tensor)
tensorList_at_s = cast2 Unmanaged.tensorList_at_s

tensorList_push_back_t ::
  ForeignPtr TensorList ->
  ForeignPtr Tensor ->
  IO (())
tensorList_push_back_t = cast2 Unmanaged.tensorList_push_back_t
