module Torch.Types.TH
  ( module Torch.Types.TH.Structs

  , C'THState, C'THNNState, CState, State(..), asState
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

  , C'THHalfStorage, C'THHalfTensor, C'THFile, C'THHalf
  ) where

import Foreign
import Foreign.C.Types
import GHC.TypeLits

import Torch.Types.TH.Structs

type C'THState = ()
type C'THNNState = C'THState
type CState = C'THState

newtype State = State { asForeign :: ForeignPtr C'THState }
  deriving (Eq, Show)
asState = State

type CAllocator   = C'THAllocator
type CDescBuff    = C'THDescBuff
type CGenerator   = C'THGenerator
newtype Generator = Generator { rng :: ForeignPtr CGenerator }
  deriving (Eq, Show)

-- data CDoubleTensor
type CInt' = CInt
type Int' = Int

-- Some type alias'
type CMaskTensor   = CByteTensor
type CIndexTensor  = CLongTensor
type CIndexStorage = CLongStorage

-- unsigned types

type CByteTensor      = C'THByteTensor
byteDynTensor         = ByteDynTensor
newtype ByteDynTensor = ByteDynTensor { byteCTensor :: ForeignPtr CByteTensor }
  deriving (Show, Eq)

type CByteStorage   = C'THByteStorage
byteStorage         = ByteStorage
newtype ByteStorage = ByteStorage { byteCStorage :: ForeignPtr CByteStorage }
  deriving (Show, Eq)


type CCharTensor      = C'THCharTensor
charDynTensor         = CharDynTensor
newtype CharDynTensor = CharDynTensor { charCTensor :: ForeignPtr CCharTensor }
  deriving (Show, Eq)

type CCharStorage   = C'THCharStorage
charStorage         = CharStorage
newtype CharStorage = CharStorage { charCStorage :: ForeignPtr CCharStorage }
  deriving (Show, Eq)


-- Signed types

type CLongTensor      = C'THLongTensor
longDynTensor         = LongDynTensor
newtype LongDynTensor = LongDynTensor { longCTensor :: ForeignPtr CLongTensor }
  deriving (Show, Eq)

type CLongStorage   = C'THLongStorage
longStorage         = LongStorage
newtype LongStorage = LongStorage { longCStorage :: ForeignPtr CLongStorage }
  deriving (Show, Eq)


type CShortTensor      = C'THShortTensor
shortDynTensor         = ShortDynTensor
newtype ShortDynTensor = ShortDynTensor { shortCTensor :: ForeignPtr CShortTensor }
  deriving (Show, Eq)

type CShortStorage   = C'THShortStorage
shortStorage         = ShortStorage
newtype ShortStorage = ShortStorage { shortCStorage :: ForeignPtr CShortStorage }
  deriving (Show, Eq)


type CIntTensor      = C'THIntTensor
intDynTensor         = IntDynTensor
newtype IntDynTensor = IntDynTensor { intCTensor :: ForeignPtr CIntTensor }
  deriving (Show, Eq)

type CIntStorage   = C'THIntStorage
intStorage         = IntStorage
newtype IntStorage = IntStorage { intCStorage :: ForeignPtr CIntStorage }
  deriving (Show, Eq)


-- Floating types

type CFloatTensor      = C'THFloatTensor
floatDynTensor         = FloatDynTensor
newtype FloatDynTensor = FloatDynTensor { floatCTensor :: ForeignPtr CFloatTensor }
  deriving (Show, Eq)

type CFloatStorage   = C'THFloatStorage
floatStorage         = FloatStorage
newtype FloatStorage = FloatStorage { floatCStorage :: ForeignPtr CFloatStorage }
  deriving (Show, Eq)


type CDoubleTensor      = C'THDoubleTensor
doubleDynTensor         = DoubleDynTensor
newtype DoubleDynTensor = DoubleDynTensor { doubleCTensor :: ForeignPtr CDoubleTensor }
  deriving (Show, Eq)

type CDoubleStorage   = C'THDoubleStorage
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

type C'THHalfStorage = ()
type C'THHalfTensor  = ()
type C'THFile = ()
type C'THHalf = Ptr ()


