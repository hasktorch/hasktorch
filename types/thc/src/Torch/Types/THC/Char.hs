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

type CTensor = CCharTensor
type CStorage = CCharStorage

type CReal = CChar
type CUReal = CUChar
type CAccReal = CLong
type HsReal = Int8
type HsUReal = Word8
type HsAccReal = Int64

hs2cReal :: HsReal -> CReal
hs2cReal = fromIntegral

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = fromIntegral

c2hsReal :: CReal -> HsReal
c2hsReal = fromIntegral

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = fromIntegral

type Storage = CharStorage
cstorage        = fst . charStorageState
storage s t     = CharStorage (s, t)
storageState    = charStorageState
storageStateRef = snd . charStorageState

type Dynamic    = CharDynamic
ctensor         = fst . charDynamicState
dynamic s t     = CharDynamic (s, t)
dynamicState    = charDynamicState
dynamicStateRef = snd . charDynamicState

newtype Tensor (ds :: [Nat]) = Tensor { asDynamic :: Dynamic }
  deriving (Show, Eq)

asStatic = Tensor


