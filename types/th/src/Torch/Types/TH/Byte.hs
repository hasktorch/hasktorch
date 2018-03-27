{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Byte where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import GHC.Word
import Torch.Types.TH
import qualified Torch.Types.TH.Random as Rand

type CAllocator = CTHAllocator
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHByteTensor
type CStorage = CTHByteStorage
type CIndexTensor = CTHLongTensor
type CIndexStorage = CTHLongStorage
type CReal = CUChar
type CAccReal = CLong
type HsReal = Word8
type HsAccReal = Word64

-- for RNG
type CGenerator = C'THGenerator
type CDoubleTensor = CTHDoubleTensor

type CMaskTensor = C'THByteTensor
type CInt' = CInt

type HsState = Ptr ()
type HsAllocator = Ptr ()
type HsDescBuff = String
type HsIndexTensor  = LongDynTensor
type HsIndexStorage = LongStorage
type HsMaskTensor   =      DynTensor
type HsGenerator    = Rand.Generator
type HsInt'         = Int

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

-- use the imports from Torch.Types.TH to break the dependency cycle
type Storage = ByteStorage
storage = byteStorage
asStorage = ByteStorage

type DynTensor = ByteDynTensor
tensor = byteTensor
asDyn = ByteDynTensor

type Tensor = ByteTensor
dynamic = byteDynamic
asStatic = ByteTensor

