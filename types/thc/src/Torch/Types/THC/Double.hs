module Torch.Types.THC.Double where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.THC

type CTensor = C'THCudaDoubleTensor
type CStorage = C'THCDoubleStorage

-- TENSOR-LAPACK ONLY
-- type CIntTensor = C'THCudaIntTensor

type CReal = CDouble
type CAccReal = CDouble
type HsReal = Double
type HsAccReal = Double

real2acc :: HsReal -> HsAccReal
real2acc = id

acc2real :: HsAccReal -> HsReal
acc2real = id

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


