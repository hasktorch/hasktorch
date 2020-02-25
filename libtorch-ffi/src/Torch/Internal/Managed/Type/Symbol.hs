
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Symbol where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Unmanaged.Type.Generator
import Torch.Internal.Unmanaged.Type.IntArray
import Torch.Internal.Unmanaged.Type.Scalar
import Torch.Internal.Unmanaged.Type.Storage
import Torch.Internal.Unmanaged.Type.Tensor
import Torch.Internal.Unmanaged.Type.TensorList
import Torch.Internal.Unmanaged.Type.TensorOptions
import Torch.Internal.Unmanaged.Type.Tuple
import Torch.Internal.Unmanaged.Type.StdString
import Torch.Internal.Unmanaged.Type.Dimname
import Torch.Internal.Unmanaged.Type.DimnameList

import qualified Torch.Internal.Unmanaged.Type.Symbol as Unmanaged



newSymbol
  :: IO (ForeignPtr Symbol)
newSymbol = cast0 Unmanaged.newSymbol





symbol_is_attr
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_attr = cast1 Unmanaged.symbol_is_attr

symbol_is_aten
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_aten = cast1 Unmanaged.symbol_is_aten

symbol_is_prim
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_prim = cast1 Unmanaged.symbol_is_prim

symbol_is_onnx
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_onnx = cast1 Unmanaged.symbol_is_onnx

symbol_is_user
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_user = cast1 Unmanaged.symbol_is_user

symbol_is_caffe2
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_caffe2 = cast1 Unmanaged.symbol_is_caffe2

symbol_is_dimname
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_dimname = cast1 Unmanaged.symbol_is_dimname



attr_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
attr_s = cast1 Unmanaged.attr_s

aten_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
aten_s = cast1 Unmanaged.aten_s

onnx_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
onnx_s = cast1 Unmanaged.onnx_s

prim_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
prim_s = cast1 Unmanaged.prim_s

user_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
user_s = cast1 Unmanaged.user_s

caffe2_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
caffe2_s = cast1 Unmanaged.caffe2_s

dimname_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
dimname_s = cast1 Unmanaged.dimname_s

