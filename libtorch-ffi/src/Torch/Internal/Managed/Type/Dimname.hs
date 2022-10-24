
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
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Type.Dimname as Unmanaged





newDimname_n
  :: ForeignPtr Dimname
  -> IO (ForeignPtr Dimname)
newDimname_n = _cast1 Unmanaged.newDimname_n

dimname_symbol
  :: ForeignPtr Dimname
  -> IO (ForeignPtr Symbol)
dimname_symbol = _cast1 Unmanaged.dimname_symbol

dimname_isBasic
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isBasic = _cast1 Unmanaged.dimname_isBasic

dimname_isWildcard
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isWildcard = _cast1 Unmanaged.dimname_isWildcard

dimname_matches_n
  :: ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (CBool)
dimname_matches_n = _cast2 Unmanaged.dimname_matches_n

fromSymbol_s
  :: ForeignPtr Symbol
  -> IO (ForeignPtr Dimname)
fromSymbol_s = _cast1 Unmanaged.fromSymbol_s

wildcard
  :: IO (ForeignPtr Dimname)
wildcard = _cast0 Unmanaged.wildcard

isValidName_s
  :: ForeignPtr StdString
  -> IO (CBool)
isValidName_s = _cast1 Unmanaged.isValidName_s

