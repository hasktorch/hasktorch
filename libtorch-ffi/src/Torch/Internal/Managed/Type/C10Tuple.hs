
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
newC10Tuple = cast0 Unmanaged.newC10Tuple





c10Tuple_empty
  :: ForeignPtr (C10Ptr IVTuple)
  -> IO (CBool)
c10Tuple_empty = cast1 Unmanaged.c10Tuple_empty

c10Tuple_size
  :: ForeignPtr (C10Ptr IVTuple)
  -> IO (CSize)
c10Tuple_size = cast1 Unmanaged.c10Tuple_size

c10Tuple_at
  :: ForeignPtr (C10Ptr IVTuple)
  -> CSize
  -> IO (ForeignPtr IValue)
c10Tuple_at = cast2 Unmanaged.c10Tuple_at

c10Tuple_push_back
  :: ForeignPtr (C10Ptr IVTuple)
  -> ForeignPtr IValue
  -> IO (())
c10Tuple_push_back = cast2 Unmanaged.c10Tuple_push_back



