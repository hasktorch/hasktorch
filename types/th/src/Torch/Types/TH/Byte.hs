{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Byte where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import GHC.Word
import Torch.Types.TH

type CStorage = CByteStorage
type CTensor = CByteTensor

type CReal = CUChar
type CAccReal = CLong
type HsReal = Word8
type HsAccReal = Word64

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

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

newtype Tensor (ds :: [Nat]) = Tensor { asDynamic :: Dynamic }
  deriving (Show, Eq)

asStatic = Tensor


