module THByteTypes
  ( CTensor
  , CStorage
  , CReal
  , CAccReal
  , HsAccReal
  , HsReal
  , hs2cReal
  , hs2cAccReal
  , c2hsReal
  , c2hsAccReal
  ) where

import Foreign.C.Types
import THTypes
import GHC.Word

type CTensor = CTHByteTensor
type CStorage = CTHByteStorage
type CReal = CChar
-- FIXME: should be:
-- type CReal = CUChar
type CAccReal = CLong
type HsReal = Word8
type HsAccReal = Word64

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

