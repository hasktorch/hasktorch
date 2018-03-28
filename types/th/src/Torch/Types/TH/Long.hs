module Torch.Types.TH.Long where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Types.TH

type CTensor = CLongTensor
type CStorage = CLongStorage
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

type Storage = LongStorage
storage = longCStorage
asStorage = LongStorage

type DynTensor = LongDynTensor
tensor = longCTensor
asDyn = LongDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


