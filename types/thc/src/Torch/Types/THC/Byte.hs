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

type CStorage = CByteStorage
type CTensor = CByteTensor

type CReal = CUChar
type CAccReal = CLong
type HsReal = Word8
type HsAccReal = Word64

-- type CDoubleTensor = C'THCudaDoubleTensor

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

type Storage = ByteStorage
storage = byteCStorage
asStorage = ByteStorage

type DynTensor = ByteDynTensor
tensor = byteCTensor
asDyn = ByteDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


