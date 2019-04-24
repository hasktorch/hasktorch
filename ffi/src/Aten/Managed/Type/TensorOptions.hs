
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.TensorOptions where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import qualified Aten.Unmanaged.Type.TensorOptions as Unmanaged



newTensorOptions
  :: DeviceType
  -> IO (ForeignPtr TensorOptions)
newTensorOptions = cast1 Unmanaged.newTensorOptions





tensorOptions_dtype
  :: ForeignPtr TensorOptions
  -> ScalarType
  -> IO (ForeignPtr TensorOptions)
tensorOptions_dtype = cast2 Unmanaged.tensorOptions_dtype

