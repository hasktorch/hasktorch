{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.TH.Int where

import Foreign.C.Types
import Foreign
import GHC.TypeLits (Nat)
import GHC.Int
import Torch.Types.TH
import qualified Torch.Types.TH.Long as Long
import qualified Torch.Types.TH.Byte as Byte
import qualified Torch.Types.TH.Random as Rand

type HsState        = Ptr ()
type HsAllocator    = Ptr ()
type HsDescBuff     = String
type HsIndexTensor  = Long.DynTensor
type HsIndexStorage = Long.Storage
type HsMaskTensor   = Byte.DynTensor
type HsGenerator    = Rand.Generator
type HsInt'         = Int


type CAllocator = CTHAllocator
type CGenerator = CTHGenerator
type CState = C'THState
type CDescBuff = C'THDescBuff
type CTensor = CTHIntTensor
type CStorage = CTHIntStorage
type CMaskTensor = CTHByteTensor
type CIndexTensor = CTHLongTensor
type CIndexStorage = CTHLongStorage

type CInt' = CInt
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


