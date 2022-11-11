
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.StdVector where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects

import qualified Torch.Internal.Unmanaged.Type.StdVector as Unmanaged



newStdVectorDouble :: IO (ForeignPtr (StdVector CDouble))
newStdVectorDouble = _cast0 Unmanaged.newStdVectorDouble

newStdVectorInt :: IO (ForeignPtr (StdVector Int64))
newStdVectorInt = _cast0 Unmanaged.newStdVectorInt

newStdVectorBool :: IO (ForeignPtr (StdVector CBool))
newStdVectorBool = _cast0 Unmanaged.newStdVectorBool





stdVectorDouble_empty :: ForeignPtr (StdVector CDouble) -> IO (CBool)
stdVectorDouble_empty = _cast1 Unmanaged.stdVectorDouble_empty

stdVectorInt_empty :: ForeignPtr (StdVector Int64) -> IO (CBool)
stdVectorInt_empty = _cast1 Unmanaged.stdVectorInt_empty

stdVectorBool_empty :: ForeignPtr (StdVector CBool) -> IO (CBool)
stdVectorBool_empty = _cast1 Unmanaged.stdVectorBool_empty

stdVectorDouble_size :: ForeignPtr (StdVector CDouble) -> IO (CSize)
stdVectorDouble_size = _cast1 Unmanaged.stdVectorDouble_size

stdVectorInt_size :: ForeignPtr (StdVector Int64) -> IO (CSize)
stdVectorInt_size = _cast1 Unmanaged.stdVectorInt_size

stdVectorBool_size :: ForeignPtr (StdVector CBool) -> IO (CSize)
stdVectorBool_size = _cast1 Unmanaged.stdVectorBool_size

stdVectorDouble_at :: ForeignPtr (StdVector CDouble) -> CSize -> IO CDouble
stdVectorDouble_at = _cast2 Unmanaged.stdVectorDouble_at

stdVectorInt_at :: ForeignPtr (StdVector Int64) -> CSize -> IO Int64
stdVectorInt_at = _cast2 Unmanaged.stdVectorInt_at

stdVectorBool_at :: ForeignPtr (StdVector CBool) -> CSize -> IO CBool
stdVectorBool_at = _cast2 Unmanaged.stdVectorBool_at

stdVectorDouble_push_back :: ForeignPtr (StdVector CDouble) -> CDouble -> IO ()
stdVectorDouble_push_back = _cast2 Unmanaged.stdVectorDouble_push_back

stdVectorInt_push_back :: ForeignPtr (StdVector Int64) -> Int64 -> IO ()
stdVectorInt_push_back = _cast2 Unmanaged.stdVectorInt_push_back

stdVectorBool_push_back :: ForeignPtr (StdVector CBool) -> CBool -> IO ()
stdVectorBool_push_back = _cast2 Unmanaged.stdVectorBool_push_back
