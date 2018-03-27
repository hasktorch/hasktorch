{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Char where

import Foreign
import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Word
import GHC.Int
import Torch.Types.TH
import qualified Torch.Types.TH.Long as Long
import qualified Torch.Types.TH.Byte as Byte
import qualified Torch.Types.TH.Random as Rand

type CAllocator = CTHAllocator
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHCharTensor
type CStorage = CTHCharStorage
type CIndexTensor = CTHLongTensor
type CIndexStorage = CTHLongStorage
type CReal = CChar
type CUReal = CUChar
type CAccReal = CLong
type HsReal = Int8
type HsUReal = Word8
type HsAccReal = Int64

type HsStorage = Storage
type HsState = Ptr ()
type HsAllocator = Ptr ()
type HsDescBuff = String
type HsIndexTensor  = Long.DynTensor
type HsIndexStorage = Long.Storage
type HsMaskTensor   = Byte.DynTensor
type HsGenerator    = Rand.Generator
type HsInt' = Int

-- for RNG
type CGenerator = C'THGenerator
type CDoubleTensor = CTHDoubleTensor

type CMaskTensor = C'THByteTensor
type CInt' = CInt

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

