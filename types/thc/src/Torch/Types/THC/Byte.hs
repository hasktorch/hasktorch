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

real2acc :: HsReal -> HsAccReal
real2acc = fromIntegral

acc2real :: HsAccReal -> HsReal
acc2real = fromIntegral

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

i2hsReal :: Integral i => i -> HsReal
i2hsReal = fromIntegral

type Storage = ByteStorage
cstorage        = snd . byteStorageState
storage         = byteStorage
storageState    = byteStorageState
storageStateRef = fst . byteStorageState

type Dynamic    = ByteDynamic
ctensor         = snd . byteDynamicState
dynamic         = byteDynamic
dynamicState    = byteDynamicState
dynamicStateRef = fst . byteDynamicState

type Tensor = ByteTensor
asDynamic = byteAsDynamic
asStatic = byteAsStatic

instance Fractional Word8
