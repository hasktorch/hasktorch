
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.IntArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.IntArray as Unmanaged



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



