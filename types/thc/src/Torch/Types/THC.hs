module Torch.Types.THC
  ( module Torch.Types.THC.Structs

  , CState, State(..), asState
  , CAllocator
  , CDescBuff
  , CGenerator, Generator(..)
  , CInt', Int'
  , CMaskTensor, CIndexTensor, CIndexStorage

  , CByteTensor, byteDynTensor, ByteDynTensor(..)
  , CByteStorage, byteStorage, ByteStorage(..)

  , CCharTensor, charDynTensor, CharDynTensor(..)
  , CCharStorage, charStorage, CharStorage(..)

  , CLongTensor, longDynTensor, LongDynTensor(..)
  , CLongStorage, longStorage, LongStorage(..)

  , CShortTensor, shortDynTensor, ShortDynTensor(..)
  , CShortStorage, shortStorage, ShortStorage(..)

  , CIntTensor, intDynTensor, IntDynTensor(..)
  , CIntStorage, intStorage, IntStorage(..)

  , CFloatTensor, floatDynTensor, FloatDynTensor(..)
  , CFloatStorage, floatStorage, FloatStorage(..)

  , CDoubleTensor, doubleDynTensor, DoubleDynTensor(..)
  , CDoubleStorage, doubleStorage, DoubleStorage(..)

  , C'THCHalfStorage, C'THCudaHalfTensor, C'THCFile, C'THCHalf
  ) where

import Foreign
import Foreign.C.Types
import GHC.TypeLits

import Torch.Types.THC.Structs

type CAllocator = ()
type CDescBuff = C'THCDescBuff

type CState = C'THCState
newtype State = State { asForeign :: ForeignPtr CState }
  deriving (Eq, Show)
asState = State

type CGenerator = C'_Generator
newtype Generator = Generator { rng :: ForeignPtr CGenerator }
  deriving (Eq, Show)

type CInt' = CLLong
type Int' = Integer

-- Some type alias'
type CMaskTensor   = CByteTensor
type CIndexTensor  = CLongTensor
type CIndexStorage = CLongStorage

-- unsigned types

type CByteTensor      = C'THCudaByteTensor
byteDynTensor         = ByteDynTensor
newtype ByteDynTensor = ByteDynTensor { byteCTensor :: ForeignPtr CByteTensor }
  deriving (Show, Eq)

type CByteStorage   = C'THCByteStorage
byteStorage         = ByteStorage
newtype ByteStorage = ByteStorage { byteCStorage :: ForeignPtr CByteStorage }
  deriving (Show, Eq)


type CCharTensor      = C'THCudaCharTensor
charDynTensor         = CharDynTensor
newtype CharDynTensor = CharDynTensor { charCTensor :: ForeignPtr CCharTensor }
  deriving (Show, Eq)

type CCharStorage   = C'THCCharStorage
charStorage         = CharStorage
newtype CharStorage = CharStorage { charCStorage :: ForeignPtr CCharStorage }
  deriving (Show, Eq)


-- Signed types

type CLongTensor      = C'THCudaLongTensor
longDynTensor         = LongDynTensor
newtype LongDynTensor = LongDynTensor { longCTensor :: ForeignPtr CLongTensor }
  deriving (Show, Eq)

type CLongStorage   = C'THCLongStorage
longStorage         = LongStorage
newtype LongStorage = LongStorage { longCStorage :: ForeignPtr CLongStorage }
  deriving (Show, Eq)


type CShortTensor      = C'THCudaShortTensor
shortDynTensor         = ShortDynTensor
newtype ShortDynTensor = ShortDynTensor { shortCTensor :: ForeignPtr CShortTensor }
  deriving (Show, Eq)

type CShortStorage   = C'THCShortStorage
shortStorage         = ShortStorage
newtype ShortStorage = ShortStorage { shortCStorage :: ForeignPtr CShortStorage }
  deriving (Show, Eq)


type CIntTensor      = C'THCudaIntTensor
intDynTensor         = IntDynTensor
newtype IntDynTensor = IntDynTensor { intCTensor :: ForeignPtr CIntTensor }
  deriving (Show, Eq)

type CIntStorage   = C'THCIntStorage
intStorage         = IntStorage
newtype IntStorage = IntStorage { intCStorage :: ForeignPtr CIntStorage }
  deriving (Show, Eq)


-- Floating types

type CFloatTensor      = C'THCudaFloatTensor
floatDynTensor         = FloatDynTensor
newtype FloatDynTensor = FloatDynTensor { floatCTensor :: ForeignPtr CFloatTensor }
  deriving (Show, Eq)

type CFloatStorage   = C'THCFloatStorage
floatStorage         = FloatStorage
newtype FloatStorage = FloatStorage { floatCStorage :: ForeignPtr CFloatStorage }
  deriving (Show, Eq)


type CDoubleTensor      = C'THCudaDoubleTensor
doubleDynTensor         = DoubleDynTensor
newtype DoubleDynTensor = DoubleDynTensor { doubleCTensor :: ForeignPtr CDoubleTensor }
  deriving (Show, Eq)

type CDoubleStorage   = C'THCDoubleStorage
doubleStorage         = DoubleStorage
newtype DoubleStorage = DoubleStorage { doubleCStorage :: ForeignPtr CDoubleStorage }
  deriving (Show, Eq)


{-
data CHalfTensor
data HalfDynTensor
halfCTensor   :: HalfDynTensor -> ForeignPtr CHalfTensor
halfDynTensor :: ForeignPtr CHalfTensor -> HalfDynTensor

data CHalfStorage
data HalfStorage
halfCStorage :: HalfStorage -> ForeignPtr CHalfStorage
halfStorage  :: ForeignPtr CHalfStorage -> HalfStorage
-}

type C'THCudaHalfTensor  = ()
type C'THCHalfStorage = ()
type C'THCFile = ()
type C'THCHalf = Ptr ()


