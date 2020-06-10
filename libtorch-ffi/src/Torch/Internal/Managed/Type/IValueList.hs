
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.IValueList where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.IValue
import Torch.Internal.Unmanaged.Type.IValueList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.IValueList as Unmanaged



newIValueList
  :: IO (ForeignPtr IValueList)
newIValueList = cast0 Unmanaged.newIValueList





ivalueList_empty
  :: ForeignPtr IValueList
  -> IO (CBool)
ivalueList_empty = cast1 Unmanaged.ivalueList_empty

ivalueList_size
  :: ForeignPtr IValueList
  -> IO (CSize)
ivalueList_size = cast1 Unmanaged.ivalueList_size

ivalueList_at
  :: ForeignPtr IValueList
  -> CSize
  -> IO (ForeignPtr IValue)
ivalueList_at = cast2 Unmanaged.ivalueList_at

ivalueList_push_back
  :: ForeignPtr IValueList
  -> ForeignPtr IValue
  -> IO (())
ivalueList_push_back = cast2 Unmanaged.ivalueList_push_back



