{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Long where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.THC


type CTensor = CTHCudaLongTensor
type CState = C'THCState
type CDescBuff = C'THCDescBuff
type CAllocator = C'THCAllocator
type CGenerator = C'THCGenerator

type CIndexTensor = C'THCudaLongTensor
type CMaskTensor = C'THCudaByteTensor
type CInt' = CLLong
type CIndexStorage = C'THCLongStorage
type CStorage = CTHCLongStorage
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

newtype Storage = Storage { storage :: ForeignPtr CStorage }
  deriving (Eq, Show)

newtype DynTensor = DynTensor { tensor :: ForeignPtr CTensor }
  deriving (Show, Eq)

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStorage = Storage
asDyn = DynTensor
asStatic = Tensor


