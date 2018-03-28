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
cstorage        = fst . doubleStorageState
storage s t     = DoubleStorage (s, t)
storageState    = doubleStorageState
storageStateRef = snd . doubleStorageState

type Dynamic    = DoubleDynamic
ctensor         = fst . doubleDynamicState
dynamic s t     = DoubleDynamic (s, t)
dynamicState    = doubleDynamicState
dynamicStateRef = snd . doubleDynamicState

newtype Tensor (ds :: [Nat]) = Tensor { asDynamic :: Dynamic }
  deriving (Show, Eq)

asStatic = Tensor


