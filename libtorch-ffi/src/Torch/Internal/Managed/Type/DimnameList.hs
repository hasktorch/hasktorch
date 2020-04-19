{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.DimnameList where

import Foreign hiding (newForeignPtr)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Concurrent
import Torch.Internal.Cast
import Torch.Internal.Class
import Torch.Internal.Type
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList
import qualified Torch.Internal.Unmanaged.Type.DimnameList as Unmanaged
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

newDimnameList ::
  IO (ForeignPtr DimnameList)
newDimnameList = cast0 Unmanaged.newDimnameList

dimnameList_empty ::
  ForeignPtr DimnameList ->
  IO (CBool)
dimnameList_empty = cast1 Unmanaged.dimnameList_empty

dimnameList_size ::
  ForeignPtr DimnameList ->
  IO (CSize)
dimnameList_size = cast1 Unmanaged.dimnameList_size

dimnameList_at_s ::
  ForeignPtr DimnameList ->
  CSize ->
  IO (ForeignPtr Dimname)
dimnameList_at_s = cast2 Unmanaged.dimnameList_at_s

dimnameList_push_back_n ::
  ForeignPtr DimnameList ->
  ForeignPtr Dimname ->
  IO (())
dimnameList_push_back_n = cast2 Unmanaged.dimnameList_push_back_n
