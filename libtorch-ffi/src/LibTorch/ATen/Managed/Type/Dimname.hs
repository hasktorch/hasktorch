
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module LibTorch.ATen.Managed.Type.Dimname where


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
import LibTorch.ATen.Unmanaged.Type.Symbol

import qualified LibTorch.ATen.Unmanaged.Type.Dimname as Unmanaged



newDimname_n
  :: ForeignPtr Dimname
  -> IO (ForeignPtr Dimname)
newDimname_n = cast1 Unmanaged.newDimname_n





dimname_symbol
  :: ForeignPtr Dimname
  -> IO (ForeignPtr Symbol)
dimname_symbol = cast1 Unmanaged.dimname_symbol

dimname_isBasic
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isBasic = cast1 Unmanaged.dimname_isBasic

dimname_isWildcard
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isWildcard = cast1 Unmanaged.dimname_isWildcard

dimname_matches_n
  :: ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (CBool)
dimname_matches_n = cast2 Unmanaged.dimname_matches_n



fromSymbol_s
  :: ForeignPtr Symbol
  -> IO (ForeignPtr Dimname)
fromSymbol_s = cast1 Unmanaged.fromSymbol_s

wildcard
  :: IO (ForeignPtr Dimname)
wildcard = cast0 Unmanaged.wildcard

isValidName_s
  :: ForeignPtr StdString
  -> IO (CBool)
isValidName_s = cast1 Unmanaged.isValidName_s

