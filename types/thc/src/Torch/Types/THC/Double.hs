{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC.Double where

import Foreign
import Foreign.C.Types
import Foreign (ForeignPtr)
import GHC.TypeLits (Nat)
import Torch.Types.THC

type CTensor = C'THCudaDoubleTensor
type CStorage = C'THCDoubleStorage

-- TENSOR-LAPACK ONLY
-- type CIntTensor = C'THCudaIntTensor

-- nn-package
type CNNState = C'THCState
type CDim = CInt
type CNNGenerator = ()

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


