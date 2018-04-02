module Torch.Types.TH
  ( module Torch.Types.TH.Structs

  , C'THState, C'THNNState, CState, State(..), asState, newCState, manageState
  , CAllocator, Allocator
  , CGenerator, Generator(..)
  , CDescBuff, DescBuff, descBuff

  -- for nn-packages
  , CNNState
  , CDim
  , CNNGenerator

  , CInt'
  , CMaskTensor, CIndexTensor, CIndexStorage
  ,  MaskDynamic,  IndexDynamic,  MaskTensor, IndexTensor, IndexStorage

  , CByteTensor, ByteDynamic(..), byteDynamic, ByteTensor(..), byteAsStatic
  , CByteStorage, ByteStorage(..), byteStorage

  , CCharTensor, CharDynamic(..), charDynamic, CharTensor(..), charAsStatic
  , CCharStorage, CharStorage(..), charStorage

  , CLongTensor, LongDynamic(..), longDynamic, LongTensor(..), longAsStatic
  , CLongStorage, LongStorage(..), longStorage

  , CShortTensor, ShortDynamic(..), shortDynamic, ShortTensor(..), shortAsStatic
  , CShortStorage, ShortStorage(..), shortStorage

  , CIntTensor, IntDynamic(..), intDynamic, IntTensor(..), intAsStatic
  , CIntStorage, IntStorage(..), intStorage

  , CFloatTensor, FloatDynamic(..), floatDynamic, FloatTensor(..), floatAsStatic
  , CFloatStorage, FloatStorage(..), floatStorage

  , CDoubleTensor, DoubleDynamic(..), doubleDynamic, DoubleTensor(..), doubleAsStatic
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

-- for nn-package
type CNNState = CState
type CDim = CLLong
type CNNGenerator = CGenerator

-- data CDoubleTensor
type CInt' = CInt
type Int' = Int

-- Some type alias'
type CMaskTensor    = CByteTensor
type CIndexTensor   = CLongTensor
type CIndexStorage  = CLongStorage
type  MaskDynamic   =  ByteDynamic
type  MaskTensor d  =  ByteTensor d
type  IndexDynamic  =  LongDynamic
type  IndexTensor d =  LongTensor d
type  IndexStorage  =  LongStorage

-- unsigned types

type CByteTensor      = C'THByteTensor
newtype ByteDynamic   = ByteDynamic { byteDynamicState :: (ForeignPtr CState, ForeignPtr CByteTensor) }
  deriving (Show, Eq)
byteDynamic = curry ByteDynamic

newtype ByteTensor (ds :: [Nat]) = ByteTensor { byteAsDynamic :: ByteDynamic }
  deriving (Show, Eq)
byteAsStatic = ByteTensor

type CByteStorage   = C'THByteStorage
newtype ByteStorage = ByteStorage { byteStorageState :: (ForeignPtr CState, ForeignPtr CByteStorage) }
  deriving (Show, Eq)
byteStorage = curry ByteStorage

type CCharTensor      = C'THCharTensor
newtype CharDynamic = CharDynamic { charDynamicState :: (ForeignPtr CState, ForeignPtr CCharTensor) }
  deriving (Show, Eq)
charDynamic = curry CharDynamic

newtype CharTensor (ds :: [Nat]) = CharTensor { charAsDynamic :: CharDynamic }
  deriving (Show, Eq)
charAsStatic = CharTensor

type CCharStorage   = C'THCharStorage
newtype CharStorage = CharStorage { charStorageState :: (ForeignPtr CState, ForeignPtr CCharStorage) }
  deriving (Show, Eq)
charStorage = curry CharStorage

-- Signed types

type CLongTensor      = C'THLongTensor
newtype LongDynamic = LongDynamic { longDynamicState :: (ForeignPtr CState, ForeignPtr CLongTensor) }
  deriving (Show, Eq)
longDynamic = curry LongDynamic

newtype LongTensor (ds :: [Nat]) = LongTensor { longAsDynamic :: LongDynamic }
  deriving (Show, Eq)
longAsStatic = LongTensor

type CLongStorage   = C'THLongStorage
newtype LongStorage = LongStorage { longStorageState :: (ForeignPtr CState, ForeignPtr CLongStorage) }
  deriving (Show, Eq)
longStorage = curry LongStorage

type CShortTensor      = C'THShortTensor
newtype ShortDynamic = ShortDynamic { shortDynamicState :: (ForeignPtr CState, ForeignPtr CShortTensor) }
  deriving (Show, Eq)
shortDynamic = curry ShortDynamic

newtype ShortTensor (ds :: [Nat]) = ShortTensor { shortAsDynamic :: ShortDynamic }
  deriving (Show, Eq)
shortAsStatic = ShortTensor

type CShortStorage   = C'THShortStorage
newtype ShortStorage = ShortStorage { shortStorageState :: (ForeignPtr CState, ForeignPtr CShortStorage) }
  deriving (Show, Eq)
shortStorage = curry ShortStorage

type CIntTensor      = C'THIntTensor
newtype IntDynamic = IntDynamic { intDynamicState :: (ForeignPtr CState, ForeignPtr CIntTensor) }
  deriving (Show, Eq)
intDynamic = curry IntDynamic

newtype IntTensor (ds :: [Nat]) = IntTensor { intAsDynamic :: IntDynamic }
  deriving (Show, Eq)
intAsStatic = IntTensor

type CIntStorage   = C'THIntStorage
newtype IntStorage = IntStorage { intStorageState :: (ForeignPtr CState, ForeignPtr CIntStorage) }
  deriving (Show, Eq)
intStorage = curry IntStorage

-- Floating types

type CFloatTensor      = C'THFloatTensor
newtype FloatDynamic = FloatDynamic { floatDynamicState :: (ForeignPtr CState, ForeignPtr CFloatTensor) }
  deriving (Show, Eq)
floatDynamic = curry FloatDynamic

newtype FloatTensor (ds :: [Nat]) = FloatTensor { floatAsDynamic :: FloatDynamic }
  deriving (Show, Eq)
floatAsStatic = FloatTensor

type CFloatStorage   = C'THFloatStorage
newtype FloatStorage = FloatStorage { floatStorageState :: (ForeignPtr CState, ForeignPtr CFloatStorage) }
  deriving (Show, Eq)
floatStorage = curry FloatStorage

type CDoubleTensor      = C'THDoubleTensor
newtype DoubleDynamic = DoubleDynamic { doubleDynamicState :: (ForeignPtr CState, ForeignPtr CDoubleTensor) }
  deriving (Show, Eq)
doubleDynamic = curry DoubleDynamic

newtype DoubleTensor (ds :: [Nat]) = DoubleTensor { doubleAsDynamic :: DoubleDynamic }
  deriving (Show, Eq)
doubleAsStatic = DoubleTensor

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


