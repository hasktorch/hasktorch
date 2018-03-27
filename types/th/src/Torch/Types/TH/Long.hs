{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Long where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Types.TH
import qualified Torch.Types.TH.Random as Rand

type HsState        = Ptr ()
type HsAllocator    = Ptr ()
type HsDescBuff     = String
type HsIndexTensor  = LongDynTensor
type HsIndexStorage = LongStorage
type HsMaskTensor   = ByteDynTensor
type HsGenerator    = Rand.Generator
type HsInt'         = Int

type CAllocator = CTHAllocator
type CGenerator = C'THGenerator
type CIndexTensor = C'THLongTensor
type CIndexStorage = C'THLongStorage
type CMaskTensor = C'THByteTensor
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHLongTensor
type CStorage = CTHLongStorage
type CReal = CLong
type CAccReal = CLong
type CInt' = CInt
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

type Storage = LongStorage
storage = longStorage
asStorage = LongStorage

type DynTensor = LongDynTensor
tensor = longTensor
asDyn = LongDynTensor

type Tensor = LongTensor
dynamic = longDynamic
asStatic = LongTensor


