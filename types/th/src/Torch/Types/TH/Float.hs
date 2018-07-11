{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Float where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import Torch.Types.TH

type CTensor = CFloatTensor
type CStorage = CFloatStorage

-- for nn-package
type CNNState = CState
type CDim = CLLong
type CNNGenerator = CGenerator

type CReal = CFloat
type CAccReal = CDouble
type HsReal = Float
type HsAccReal = Double

hs2cReal :: HsReal -> CReal
hs2cReal = realToFrac

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = realToFrac

real2acc :: HsReal -> HsAccReal
real2acc = realToFrac

acc2real :: HsAccReal -> HsReal
acc2real = realToFrac

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


