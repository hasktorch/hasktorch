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
storage = charCStorage
asStorage = CharStorage

type DynTensor = CharDynTensor
tensor = charCTensor
asDyn = CharDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


