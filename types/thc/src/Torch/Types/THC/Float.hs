module Torch.Types.THC.Float where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import Torch.Types.THC

type CTensor = CFloatTensor
type CStorage = CFloatStorage

type CReal = CFloat
type CAccReal = CDouble
type HsReal = Float
type HsAccReal = Double

hs2cReal :: HsReal -> CReal
hs2cReal = realToFrac

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = realToFrac

c2hsReal :: CReal -> HsReal
c2hsReal = realToFrac

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = realToFrac

type Storage = FloatStorage
cstorage        = snd . floatStorageState
storage         = floatStorage
storageState    = floatStorageState
storageStateRef = fst . floatStorageState

type Dynamic    = FloatDynamic
ctensor         = snd . floatDynamicState
dynamic         = floatDynamic
dynamicState    = floatDynamicState
dynamicStateRef = fst . floatDynamicState

type Tensor = FloatTensor
asDynamic = floatAsDynamic
asStatic = floatAsStatic


