module Torch.Types.TH.Double where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import Torch.Types.TH

type CTensor = CDoubleTensor
type CStorage = CDoubleStorage

type CReal = CDouble
type CAccReal = CDouble
type HsReal = Double
type HsAccReal = Double

real2acc :: HsReal -> HsAccReal
real2acc = realToFrac

acc2real :: HsAccReal -> HsReal
acc2real = realToFrac

hs2cReal :: HsReal -> CReal
hs2cReal = realToFrac

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = realToFrac

c2hsReal :: CReal -> HsReal
c2hsReal = realToFrac

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = realToFrac

type Storage = DoubleStorage
cstorage        = snd . doubleStorageState
storage         = doubleStorage
storageState    = doubleStorageState
storageStateRef = fst . doubleStorageState

type Dynamic    = DoubleDynamic
ctensor         = snd . doubleDynamicState
dynamic         = doubleDynamic
dynamicState    = doubleDynamicState
dynamicStateRef = fst . doubleDynamicState

type Tensor = DoubleTensor
asDynamic = doubleAsDynamic
asStatic = doubleAsStatic


