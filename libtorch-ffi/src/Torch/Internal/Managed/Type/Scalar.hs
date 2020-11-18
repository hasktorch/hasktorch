
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Scalar where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Scalar as Unmanaged





newScalar
  :: IO (ForeignPtr Scalar)
newScalar = cast0 Unmanaged.newScalar

newScalar_i
  :: CInt
  -> IO (ForeignPtr Scalar)
newScalar_i = cast1 Unmanaged.newScalar_i

newScalar_d
  :: CDouble
  -> IO (ForeignPtr Scalar)
newScalar_d = cast1 Unmanaged.newScalar_d

newScalar_b
  :: CBool
  -> IO (ForeignPtr Scalar)
newScalar_b = cast1 Unmanaged.newScalar_b

newScalar_f
  :: CFloat
  -> IO (ForeignPtr Scalar)
newScalar_f = cast1 Unmanaged.newScalar_f

