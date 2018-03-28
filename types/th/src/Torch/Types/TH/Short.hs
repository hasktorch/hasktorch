{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Short where

import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.TH

type CTensor = CShortTensor
type CStorage = CShortStorage

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

type Storage = ShortStorage
storage = shortCStorage
asStorage = ShortStorage

type DynTensor = ShortDynTensor
tensor = shortCTensor
asDyn = ShortDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


