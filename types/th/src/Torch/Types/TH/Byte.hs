{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Byte where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Word
import Torch.Types.TH

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

type CMaskTensor = C'THByteTensor
type CInt' = CInt

-- type CByteTensor = C'THByteTensor
-- type CCharTensor = C'THCharTensor
-- type CShortTensor = C'THShortTensor
-- type CIntTensor = C'THIntTensor
-- type CLongTensor = C'THLongTensor
-- type CFloatTensor = C'THFloatTensor
-- type CDoubleTensor = C'THDoubleTensor
-- type CHalfTensor = C'THHalfTensor

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

