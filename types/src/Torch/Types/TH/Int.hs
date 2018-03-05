{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Int
  ( CTensor
  , CStorage
  , CReal
  , CAccReal
  , HsAccReal
  , HsReal
  , hs2cReal
  , hs2cAccReal
  , c2hsReal
  , c2hsAccReal
  , Tensor(..)
  , DynTensor(..)
  , Storage(..)
  , asStorage
  , asDyn
  , asStatic
  ) where

import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.TH


type CTensor = CTHIntTensor
type CStorage = CTHIntStorage
type CReal = CInt
type CAccReal = CLong
type HsReal = Int32
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


