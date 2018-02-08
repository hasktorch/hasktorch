module THIntTypes
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
import GHC.Int

type CTensor = CTHIntTensor
type CStorage = CTHIntStorage
type CReal = CInt
type CAccReal = CLong
type HsReal = Int32
type HsAccReal = Int64

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

