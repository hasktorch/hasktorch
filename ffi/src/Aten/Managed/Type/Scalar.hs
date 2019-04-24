
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.Scalar where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import qualified Aten.Unmanaged.Type.Scalar as Unmanaged



newScalar
  :: IO (ForeignPtr Scalar)
newScalar = cast0 Unmanaged.newScalar

newScalar_int
  :: CInt
  -> IO (ForeignPtr Scalar)
newScalar_int = cast1 Unmanaged.new_intScalar

newScalar_double
  :: CDouble
  -> IO (ForeignPtr Scalar)
newScalar_double = cast1 Unmanaged.new_doubleScalar





