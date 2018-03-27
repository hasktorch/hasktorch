{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Byte where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import GHC.Word
import Torch.Types.TH (C'THLongStorage, C'THLongTensor)
import Torch.Types.THC
import qualified Torch.Types.THC.Long as Long

type CState = C'THCState
type CDescBuff = C'THCDescBuff
type CAllocator = C'THCAllocator
type CTensor = CTHCudaByteTensor
type CStorage = CTHCByteStorage
type CIndexTensor = C'THCudaLongTensor
type CIndexStorage = C'THCLongStorage
type CReal = CUChar
type CAccReal = CLong
type HsReal = Word8
type HsAccReal = Word64

type CDoubleTensor = C'THCudaDoubleTensor
type CGenerator = C'THCGenerator

type CMaskTensor = C'THCudaByteTensor
type CInt' = CLLong

type HsState        = CudaState
type HsGenerator    = CudaGenerator
type HsAllocator    = Ptr ()
type HsDescBuff     = String
type HsIndexTensor  = Long.DynTensor
type HsIndexStorage = Long.Storage
type HsMaskTensor   = DynTensor
type HsInt'         = Int

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

asStorage = Storage

newtype DynTensor = DynTensor { tensor :: ForeignPtr CTensor }
  deriving (Show, Eq)

asDyn = DynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor

