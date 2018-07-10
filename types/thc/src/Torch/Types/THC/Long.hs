{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Long where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import Torch.Types.THC

type CTensor = C'THCudaLongTensor
type CStorage = C'THCLongStorage

type CReal = CLong
type CAccReal = CLong
type HsReal = Integer
type HsAccReal = Integer

real2acc :: HsReal -> HsAccReal
real2acc = id

acc2real :: HsAccReal -> HsReal
acc2real = id

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

-- instance Fractional Integer
