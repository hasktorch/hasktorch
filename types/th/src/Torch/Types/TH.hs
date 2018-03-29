module Torch.Types.TH
  ( module Torch.Types.TH.Structs

  , C'THState, C'THNNState, CState, State(..), asState, newCState, manageState
  , CAllocator, Allocator
  , CGenerator, Generator(..)
  , CDescBuff, DescBuff, descBuff

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

  , C'THHalfTensor, C'THHalfStorage, C'THFile, C'THHalf
  ) where

import Foreign
import Foreign.C.Types
import GHC.TypeLits
import Data.Char (chr)

import Torch.Types.TH.Structs

type CDescBuff = C'THDescBuff
type DescBuff = String

descBuff :: Ptr CDescBuff -> IO DescBuff
descBuff p = (map (chr . fromIntegral) . c'THDescBuff'str) <$> peek p

newCState :: IO (Ptr C'THState)
newCState = pure nullPtr

type C'THState = ()
type C'THNNState = C'THState
type CState = C'THState
newtype State = State { asForeign :: ForeignPtr C'THState }
  deriving (Eq, Show)
asState = State

manageState :: Ptr C'THState -> IO (ForeignPtr C'THState)
manageState = newForeignPtr nullFunPtr

type CAllocator   = C'THAllocator
newtype Allocator = Allocator { callocator :: ForeignPtr CAllocator }
  deriving (Eq, Show)

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
type  MaskTensor   =  ByteDynamic
type  IndexTensor  =  LongDynamic
type  IndexStorage =  LongStorage

-- unsigned types

type CByteTensor      = C'THByteTensor
newtype ByteDynamic   = ByteDynamic { byteDynamicState :: (ForeignPtr CState, ForeignPtr CByteTensor) }
  deriving (Show, Eq)
byteDynamic = curry ByteDynamic

type CByteStorage   = C'THByteStorage
newtype ByteStorage = ByteStorage { byteStorageState :: (ForeignPtr CState, ForeignPtr CByteStorage) }
  deriving (Show, Eq)
byteStorage = curry ByteStorage

type CCharTensor      = C'THCharTensor
newtype CharDynamic = CharDynamic { charDynamicState :: (ForeignPtr CState, ForeignPtr CCharTensor) }
  deriving (Show, Eq)
charDynamic = curry CharDynamic

type CCharStorage   = C'THCharStorage
newtype CharStorage = CharStorage { charStorageState :: (ForeignPtr CState, ForeignPtr CCharStorage) }
  deriving (Show, Eq)
charStorage = curry CharStorage

-- Signed types

type CLongTensor      = C'THLongTensor
newtype LongDynamic = LongDynamic { longDynamicState :: (ForeignPtr CState, ForeignPtr CLongTensor) }
  deriving (Show, Eq)
longDynamic = curry LongDynamic

type CLongStorage   = C'THLongStorage
newtype LongStorage = LongStorage { longStorageState :: (ForeignPtr CState, ForeignPtr CLongStorage) }
  deriving (Show, Eq)
longStorage = curry LongStorage

type CShortTensor      = C'THShortTensor
newtype ShortDynamic = ShortDynamic { shortDynamicState :: (ForeignPtr CState, ForeignPtr CShortTensor) }
  deriving (Show, Eq)
shortDynamic = curry ShortDynamic

type CShortStorage   = C'THShortStorage
newtype ShortStorage = ShortStorage { shortStorageState :: (ForeignPtr CState, ForeignPtr CShortStorage) }
  deriving (Show, Eq)
shortStorage = curry ShortStorage

type CIntTensor      = C'THIntTensor
newtype IntDynamic = IntDynamic { intDynamicState :: (ForeignPtr CState, ForeignPtr CIntTensor) }
  deriving (Show, Eq)
intDynamic = curry IntDynamic

type CIntStorage   = C'THIntStorage
newtype IntStorage = IntStorage { intStorageState :: (ForeignPtr CState, ForeignPtr CIntStorage) }
  deriving (Show, Eq)
intStorage = curry IntStorage

-- Floating types

type CFloatTensor      = C'THFloatTensor
newtype FloatDynamic = FloatDynamic { floatDynamicState :: (ForeignPtr CState, ForeignPtr CFloatTensor) }
  deriving (Show, Eq)
floatDynamic = curry FloatDynamic

type CFloatStorage   = C'THFloatStorage
newtype FloatStorage = FloatStorage { floatStorageState :: (ForeignPtr CState, ForeignPtr CFloatStorage) }
  deriving (Show, Eq)
floatStorage = curry FloatStorage

type CDoubleTensor      = C'THDoubleTensor
newtype DoubleDynamic = DoubleDynamic { doubleDynamicState :: (ForeignPtr CState, ForeignPtr CDoubleTensor) }
  deriving (Show, Eq)
doubleDynamic = curry DoubleDynamic

type CDoubleStorage   = C'THDoubleStorage
newtype DoubleStorage = DoubleStorage { doubleStorageState :: (ForeignPtr CState, ForeignPtr CDoubleStorage) }
  deriving (Show, Eq)
doubleStorage = curry DoubleStorage

{-
data CHalfTensor
data HalfDynTensor
halfCTensor   :: HalfDynTensor -> ForeignPtr CHalfTensor
halfDynTensor :: ForeignPtr CHalfTensor -> HalfDynTensor

-}

type C'THHalfTensor  = ()
type C'THHalfStorage  = ()
type C'THFile = ()
type C'THHalf = Ptr ()


