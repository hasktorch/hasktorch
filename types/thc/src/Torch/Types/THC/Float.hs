{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
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
cstorage        = fst . floatStorageState
storage s t     = FloatStorage (s, t)
storageState    = floatStorageState
storageStateRef = snd . floatStorageState

type Dynamic    = FloatDynamic
ctensor         = fst . floatDynamicState
dynamic s t     = FloatDynamic (s, t)
dynamicState    = floatDynamicState
dynamicStateRef = snd . floatDynamicState

newtype Tensor (ds :: [Nat]) = Tensor { asDynamic :: Dynamic }
  deriving (Show, Eq)

asStatic = Tensor


