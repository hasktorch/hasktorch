{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.Internal.Managed.Type.IValueList where

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
import Torch.Internal.Unmanaged.Type.IValue
import Torch.Internal.Unmanaged.Type.IValueList
import qualified Torch.Internal.Unmanaged.Type.IValueList as Unmanaged
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple

newIValueList ::
  IO (ForeignPtr IValueList)
newIValueList = cast0 Unmanaged.newIValueList

ivalueList_empty ::
  ForeignPtr IValueList ->
  IO (CBool)
ivalueList_empty = cast1 Unmanaged.ivalueList_empty

ivalueList_size ::
  ForeignPtr IValueList ->
  IO (CSize)
ivalueList_size = cast1 Unmanaged.ivalueList_size

ivalueList_at ::
  ForeignPtr IValueList ->
  CSize ->
  IO (ForeignPtr IValue)
ivalueList_at = cast2 Unmanaged.ivalueList_at

ivalueList_push_back ::
  ForeignPtr IValueList ->
  ForeignPtr IValue ->
  IO (())
ivalueList_push_back = cast2 Unmanaged.ivalueList_push_back
