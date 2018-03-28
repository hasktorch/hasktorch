module Torch.Types.THC
  ( module Torch.Types.THC.Structs

  , CState, State(..), asState
  , CAllocator, Allocator(..)
  , CDescBuff, DescBuff
  , CGenerator, Generator(..)
  , CInt'
  , CMaskTensor, CIndexTensor, CIndexStorage
  ,  MaskTensor,  IndexTensor,  IndexStorage

  , CByteTensor, ByteDynamic(..), byteDynamic
  , CByteStorage, ByteStorage(..), byteStorage

  , CCharTensor, CharDynamic(..), charDynamic
  , CCharStorage, CharStorage(..), charStorage

  , CLongTensor, LongDynamic(..), longDynamic
  , CLongStorage, LongStorage(..), longStorage

  , CShortTensor, ShortDynamic(..), shortDynamic
  , CShortStorage, ShortStorage(..), shortStorage

  , CIntTensor, IntDynamic(..), intDynamic
  , CIntStorage, IntStorage(..), intStorage

  , CFloatTensor, FloatDynamic(..), floatDynamic
  , CFloatStorage, FloatStorage(..), floatStorage

  , CDoubleTensor, DoubleDynamic(..), doubleDynamic
  , CDoubleStorage, DoubleStorage(..), doubleStorage

  , C'THCHalfStorage, C'THCudaHalfTensor, C'THCFile, C'THCHalf
  ) where

import Foreign
import Foreign.C.Types
import GHC.TypeLits

import Torch.Types.THC.Structs

type CAllocator = ()
type  Allocator = ()

type CDescBuff = C'THCDescBuff
type  DescBuff = String

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

type  MaskTensor   = ByteDynamic
type  IndexTensor  = LongDynamic
type  IndexStorage = LongStorage

-- unsigned types

type CByteTensor      = C'THCudaByteTensor
newtype ByteDynamic   = ByteDynamic { byteDynamicState :: (ForeignPtr CState, ForeignPtr CByteTensor) }
  deriving (Show, Eq)
byteDynamic = curry ByteDynamic

type CByteStorage   = C'THCByteStorage
newtype ByteStorage = ByteStorage { byteStorageState :: (ForeignPtr CState, ForeignPtr CByteStorage) }
  deriving (Show, Eq)
byteStorage = curry ByteStorage

type CCharTensor      = C'THCudaCharTensor
newtype CharDynamic = CharDynamic { charDynamicState :: (ForeignPtr CState, ForeignPtr CCharTensor) }
  deriving (Show, Eq)
charDynamic = curry CharDynamic

type CCharStorage   = C'THCCharStorage
newtype CharStorage = CharStorage { charStorageState :: (ForeignPtr CState, ForeignPtr CCharStorage) }
  deriving (Show, Eq)
charStorage = curry CharStorage

-- Signed types

type CLongTensor      = C'THCudaLongTensor
newtype LongDynamic = LongDynamic { longDynamicState :: (ForeignPtr CState, ForeignPtr CLongTensor) }
  deriving (Show, Eq)
longDynamic = curry LongDynamic

type CLongStorage   = C'THCLongStorage
newtype LongStorage = LongStorage { longStorageState :: (ForeignPtr CState, ForeignPtr CLongStorage) }
  deriving (Show, Eq)
longStorage = curry LongStorage

type CShortTensor      = C'THCudaShortTensor
newtype ShortDynamic = ShortDynamic { shortDynamicState :: (ForeignPtr CState, ForeignPtr CShortTensor) }
  deriving (Show, Eq)
shortDynamic = curry ShortDynamic

type CShortStorage   = C'THCShortStorage
newtype ShortStorage = ShortStorage { shortStorageState :: (ForeignPtr CState, ForeignPtr CShortStorage) }
  deriving (Show, Eq)
shortStorage = curry ShortStorage

type CIntTensor      = C'THCudaIntTensor
newtype IntDynamic = IntDynamic { intDynamicState :: (ForeignPtr CState, ForeignPtr CIntTensor) }
  deriving (Show, Eq)
intDynamic = curry IntDynamic

type CIntStorage   = C'THCIntStorage
newtype IntStorage = IntStorage { intStorageState :: (ForeignPtr CState, ForeignPtr CIntStorage) }
  deriving (Show, Eq)
intStorage = curry IntStorage

-- Floating types

type CFloatTensor      = C'THCudaFloatTensor
newtype FloatDynamic = FloatDynamic { floatDynamicState :: (ForeignPtr CState, ForeignPtr CFloatTensor) }
  deriving (Show, Eq)
floatDynamic = curry FloatDynamic

type CFloatStorage   = C'THCFloatStorage
newtype FloatStorage = FloatStorage { floatStorageState :: (ForeignPtr CState, ForeignPtr CFloatStorage) }
  deriving (Show, Eq)
floatStorage = curry FloatStorage

type CDoubleTensor      = C'THCudaDoubleTensor
newtype DoubleDynamic = DoubleDynamic { doubleDynamicState :: (ForeignPtr CState, ForeignPtr CDoubleTensor) }
  deriving (Show, Eq)
doubleDynamic = curry DoubleDynamic

type CDoubleStorage   = C'THCDoubleStorage
newtype DoubleStorage = DoubleStorage { doubleStorageState :: (ForeignPtr CState, ForeignPtr CDoubleStorage) }
  deriving (Show, Eq)
doubleStorage = curry DoubleStorage


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


