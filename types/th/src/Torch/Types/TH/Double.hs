{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Double where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.TH

type CAllocator = CTHAllocator
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHDoubleTensor
type CStorage = CTHDoubleStorage
type CReal = CDouble
type CAccReal = CDouble
type HsReal = Double
type HsAccReal = Double

hs2cReal :: HsReal -> CReal
hs2cReal = realToFrac

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = realToFrac

c2hsReal :: CReal -> HsReal
c2hsReal = realToFrac

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = realToFrac

newtype Storage = Storage { storage :: ForeignPtr CStorage }
  deriving (Eq, Show)

newtype DynTensor = DynTensor { tensor :: ForeignPtr CTensor }
  deriving (Show, Eq)

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStorage = Storage
asDyn = DynTensor
asStatic = Tensor


