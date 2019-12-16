
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Managed.Type.Dimname where


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
import Torch.Internal.Unmanaged.Type.Symbol

import qualified Torch.Internal.Unmanaged.Type.Dimname as Unmanaged



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

