{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Short
  ( CTensor
  , CState
  , CStorage
  , CReal
  , CAccReal
  , HsAccReal
  , HsReal
  , hs2cReal
  , hs2cAccReal
  , c2hsReal
  , c2hsAccReal
  , Storage(..)
  , DynTensor(..)
  , Tensor(..)
  , asStorage
  , asDyn
  , asStatic
  ) where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.THC

type CTensor = CTHCudaShortTensor
type CState = ()
type CStorage = CTHCShortStorage
type CReal = CShort
type CAccReal = CLong
type HsReal = Int16
type HsAccReal = Int64

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


