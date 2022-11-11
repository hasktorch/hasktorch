
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.C10Tuple where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.C10Tuple as Unmanaged



newC10Tuple
  :: IO (ForeignPtr (C10Ptr IVTuple))
newC10Tuple = _cast0 Unmanaged.newC10Tuple

newC10Tuple_tuple
  :: ForeignPtr IValueList
  -> IO (ForeignPtr (C10Ptr IVTuple))
newC10Tuple_tuple  = _cast1 Unmanaged.newC10Tuple_tuple

c10Tuple_empty
  :: ForeignPtr (C10Ptr IVTuple)
  -> IO (CBool)
c10Tuple_empty = _cast1 Unmanaged.c10Tuple_empty

c10Tuple_size
  :: ForeignPtr (C10Ptr IVTuple)
  -> IO (CSize)
c10Tuple_size = _cast1 Unmanaged.c10Tuple_size

c10Tuple_at
  :: ForeignPtr (C10Ptr IVTuple)
  -> CSize
  -> IO (ForeignPtr IValue)
c10Tuple_at = _cast2 Unmanaged.c10Tuple_at
