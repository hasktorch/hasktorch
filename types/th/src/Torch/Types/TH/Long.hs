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
cstorage        = snd . longStorageState
storage         = longStorage
storageState    = longStorageState
storageStateRef = fst . longStorageState

type Dynamic    = LongDynamic
ctensor         = snd . longDynamicState
dynamic         = longDynamic
dynamicState    = longDynamicState
dynamicStateRef = fst . longDynamicState

type Tensor = LongTensor
asDynamic = longAsDynamic
asStatic = longAsStatic


