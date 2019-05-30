
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.Scalar where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.SparseTensorRef
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple

import qualified ATen.Unmanaged.Type.Scalar as Unmanaged



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







