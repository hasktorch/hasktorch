{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Char where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import GHC.Word
import GHC.Int
import Torch.Types.TH (C'THLongStorage, C'THLongTensor)
import Torch.Types.THC
import qualified Torch.Types.THC.Byte as Byte
import qualified Torch.Types.THC.Long as Long

type CState = C'THCState
type CDescBuff = C'THCDescBuff
type CAllocator = C'THCState
type CTensor = CTHCudaCharTensor
type CStorage = CTHCCharStorage
type CIndexTensor = C'THCudaLongTensor
type CIndexStorage = C'THCLongStorage
type CReal = CChar
type CUReal = CUChar
type CAccReal = CLong
type HsReal = Int8
type HsUReal = Word8
type HsAccReal = Int64

type CGenerator = CTHCGenerator
type CDoubleTensor = C'THCudaDoubleTensor

type CMaskTensor = C'THCudaByteTensor
type CInt' = CLLong

type HsState        = CudaState
type HsGenerator    = CudaGenerator
type HsAllocator    = Ptr ()
type HsDescBuff     = String
type HsIndexTensor  = Long.DynTensor
type HsIndexStorage = Long.Storage
type HsMaskTensor   = Byte.DynTensor
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

