{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Short where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.THC

type CTensor = C'THCudaShortTensor
type CStorage = C'THCShortStorage

type CReal = CShort
type CAccReal = CLong
type HsReal = Int16
type HsAccReal = Int64


hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

newtype Storage = Storage { storage :: ForeignPtr CStorage }
  deriving (Eq, Show)


newtype DynTensor = DynTensor { tensor :: ForeignPtr CTensor }
  deriving (Show, Eq)


newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStorage = Storage
asDyn = DynTensor
asStatic = Tensor


