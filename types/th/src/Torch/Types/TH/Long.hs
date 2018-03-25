{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Long where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.TH


type CAllocator = CTHAllocator
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHLongTensor
type CStorage = CTHLongStorage
type CReal = CLong
type CAccReal = CLong
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

newtype Storage = Storage { storage :: ForeignPtr CStorage }
  deriving (Eq, Show)

newtype DynTensor = DynTensor { tensor :: ForeignPtr CTensor }
  deriving (Show, Eq)

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStorage = Storage
asDyn = DynTensor
asStatic = Tensor


