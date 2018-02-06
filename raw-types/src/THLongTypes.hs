module THLongTypes
  ( CTensor
  , CStorage
  , CReal
  , CAccReal
  , HsAccReal
  , HsReal
  ) where

import Foreign.C.Types
import THTypes

type CTensor = THLongTensor
type CStorage = THLongStorage
type CReal = CLong
type CAccReal = CLong
type HsReal = Integer
type HsAccReal = Integer

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

