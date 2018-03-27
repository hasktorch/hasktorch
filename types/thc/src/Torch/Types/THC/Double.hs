{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Double where

import Foreign
import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.THC

type HsState        = CudaState
type HsGenerator    = CudaGenerator
type HsAllocator    = Ptr ()
type HsDescBuff     = String
type HsIndexTensor  = LongDynTensor
type HsIndexStorage = LongStorage
type HsMaskTensor   = ByteDynTensor
type HsInt'         = Int

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

-- nn-package
type CNNState = C'THCState
type CDim = CInt
type CNNGenerator = ()

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


