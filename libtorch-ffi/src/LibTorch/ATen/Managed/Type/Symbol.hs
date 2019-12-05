
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.Symbol where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import LibTorch.ATen.Type
import LibTorch.ATen.Class
import LibTorch.ATen.Cast
import LibTorch.ATen.Unmanaged.Type.Generator
import LibTorch.ATen.Unmanaged.Type.IntArray
import LibTorch.ATen.Unmanaged.Type.Scalar
import LibTorch.ATen.Unmanaged.Type.Storage
import LibTorch.ATen.Unmanaged.Type.Tensor
import LibTorch.ATen.Unmanaged.Type.TensorList
import LibTorch.ATen.Unmanaged.Type.TensorOptions
import LibTorch.ATen.Unmanaged.Type.Tuple
import LibTorch.ATen.Unmanaged.Type.StdString
import LibTorch.ATen.Unmanaged.Type.Dimname
import LibTorch.ATen.Unmanaged.Type.DimnameList

import qualified LibTorch.ATen.Unmanaged.Type.Symbol as Unmanaged



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

