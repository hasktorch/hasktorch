{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Double where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.THC

type CTensor = CTHCudaDoubleTensor
type CStorage = CTHCDoubleStorage

type CInt' = CLLong
type CState = C'THCState
type CDescBuff = C'THCDescBuff
type CGenerator = C'THCGenerator
type CAllocator = C'THCAllocator
type CMaskTensor = C'THCudaByteTensor
type CIndexTensor = C'THCudaLongTensor
type CIndexStorage = C'THCLongStorage

-- TENSOR-LAPACK ONLY
-- type CIntTensor = C'THCudaIntTensor

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


