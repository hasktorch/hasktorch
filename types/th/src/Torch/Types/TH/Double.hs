{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Double where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import Torch.Types.TH

type CTensor = CDoubleTensor
type CStorage = CDoubleStorage

-- for nn-package
type CNNState = CState
type CDim = CLLong
type CNNGenerator = CGenerator

type CReal = CDouble
type CAccReal = CDouble
type HsReal = Double
type HsAccReal = Double

hs2cReal :: HsReal -> CReal
hs2cReal = realToFrac

hs2cAccReal :: HsAccReal -> CAccReal
hs2cAccReal = realToFrac

c2hsReal :: CReal -> HsReal
c2hsReal = realToFrac

c2hsAccReal :: CAccReal -> HsAccReal
c2hsAccReal = realToFrac

type Storage = DoubleStorage
storage = doubleCStorage
asStorage = DoubleStorage

type DynTensor = DoubleDynTensor
tensor = doubleCTensor
asDyn = DoubleDynTensor

newtype Tensor (ds :: [Nat]) = Tensor { dynamic :: DynTensor }
  deriving (Show, Eq)

asStatic = Tensor


