
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
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Symbol as Unmanaged





newSymbol
  :: IO (ForeignPtr Symbol)
newSymbol = _cast0 Unmanaged.newSymbol

symbol_is_attr
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_attr = _cast1 Unmanaged.symbol_is_attr

symbol_is_aten
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_aten = _cast1 Unmanaged.symbol_is_aten

symbol_is_prim
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_prim = _cast1 Unmanaged.symbol_is_prim

symbol_is_onnx
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_onnx = _cast1 Unmanaged.symbol_is_onnx

symbol_is_user
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_user = _cast1 Unmanaged.symbol_is_user

symbol_is_caffe2
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_caffe2 = _cast1 Unmanaged.symbol_is_caffe2

symbol_is_dimname
  :: ForeignPtr Symbol
  -> IO (CBool)
symbol_is_dimname = _cast1 Unmanaged.symbol_is_dimname

symbol_toUnqualString
  :: ForeignPtr Symbol
  -> IO (ForeignPtr StdString)
symbol_toUnqualString = _cast1 Unmanaged.symbol_toUnqualString

symbol_toQualString
  :: ForeignPtr Symbol
  -> IO (ForeignPtr StdString)
symbol_toQualString = _cast1 Unmanaged.symbol_toQualString

symbol_toDisplayString
  :: ForeignPtr Symbol
  -> IO (ForeignPtr StdString)
symbol_toDisplayString = _cast1 Unmanaged.symbol_toDisplayString

attr_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
attr_s = _cast1 Unmanaged.attr_s

aten_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
aten_s = _cast1 Unmanaged.aten_s

onnx_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
onnx_s = _cast1 Unmanaged.onnx_s

prim_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
prim_s = _cast1 Unmanaged.prim_s

user_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
user_s = _cast1 Unmanaged.user_s

caffe2_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
caffe2_s = _cast1 Unmanaged.caffe2_s

dimname_s
  :: ForeignPtr StdString
  -> IO (ForeignPtr Symbol)
dimname_s = _cast1 Unmanaged.dimname_s

