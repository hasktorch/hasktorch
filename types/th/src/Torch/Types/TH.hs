{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Torch.Types.TH
  ( module Torch.Types.TH.Structs

  , C'THState, C'THNNState, CState, State(..), asState, newCState, manageState
  , CAllocator, Allocator
  , CGenerator, Generator(..), generatorToRng, Seed(..)
  , CDescBuff, DescBuff, descBuff

  -- for nn-packages
  , CNNState
  , CDim
  , CNNGenerator

  , CInt'
  , CMaskTensor, CIndexTensor, CIndexStorage
  ,  MaskDynamic,  IndexDynamic,  MaskTensor, IndexTensor, IndexStorage


  -- * Unsigned types
  , CByteTensor, ByteDynamic(..), byteDynamic, ByteTensor(..), byteAsStatic
  , CByteStorage, ByteStorage(..), byteStorage

  , CCharTensor, CharDynamic(..), charDynamic, CharTensor(..), charAsStatic
  , CCharStorage, CharStorage(..), charStorage

  -- * Signed types
  , CLongTensor, LongDynamic(..), longDynamic, LongTensor(..), longAsStatic
  , CLongStorage, LongStorage(..), longStorage

  , CShortTensor, ShortDynamic(..), shortDynamic, ShortTensor(..), shortAsStatic
  , CShortStorage, ShortStorage(..), shortStorage

  , CIntTensor, IntDynamic(..), intDynamic, IntTensor(..), intAsStatic
  , CIntStorage, IntStorage(..), intStorage

  -- * Floating types
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

foreign import ccall "&free_CTHState" state_free :: FunPtr (Ptr C'THState -> IO ())

manageState :: Ptr C'THState -> IO (ForeignPtr C'THState)
manageState = newForeignPtr state_free

type CAllocator   = C'THAllocator
newtype Allocator = Allocator { callocator :: ForeignPtr CAllocator }
  deriving (Eq, Show)
type CGenerator   = C'THGenerator -- ^ Backpack type alias for TH's CPU generator
newtype Generator = Generator { rng :: ForeignPtr CGenerator }
  deriving (Eq, Show)
-- ^ Representation of a CPU-bound random number generator

generatorToRng :: ForeignPtr CGenerator -> Generator
generatorToRng = Generator

-- | Representation of a CPU-bound random seed
newtype Seed = Seed { unSeed :: Word64 }
  deriving (Bounded, Enum, Eq, Integral, Num, Ord, Read, Real, Show)

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
type  MaskTensor    =  ByteTensor
type  IndexDynamic  =  LongDynamic
type  IndexTensor   =  LongTensor
type  IndexStorage  =  LongStorage

-- | A C-level representation of the Byte Tensor type. These need to be wrapped in 'Ptr'
type CByteTensor      = C'THByteTensor

-- | A memory-managed representation of TH's Byte Tensor type. These carry a reference to the 'CState'
newtype ByteDynamic   = ByteDynamic { byteDynamicState :: (ForeignPtr CState, ForeignPtr CByteTensor) }
  deriving (Eq)
-- | smart constructor for 'ByteDynamic'.
byteDynamic = curry ByteDynamic

-- | A newtype wrapper around 'ByteDynamic' which imbues a 'ByteDynamic' with static tensor dimensions.
newtype ByteTensor (ds :: [Nat]) = ByteTensor { byteAsDynamic :: ByteDynamic }
  deriving (Eq)
-- | smart constructor for 'ByteTensor'.
byteAsStatic = ByteTensor

type CByteStorage   = C'THByteStorage
newtype ByteStorage = ByteStorage { byteStorageState :: (ForeignPtr CState, ForeignPtr CByteStorage) }
  deriving (Eq)
byteStorage = curry ByteStorage

type CCharTensor      = C'THCharTensor
newtype CharDynamic = CharDynamic { charDynamicState :: (ForeignPtr CState, ForeignPtr CCharTensor) }
  deriving (Eq)
charDynamic = curry CharDynamic

newtype CharTensor (ds :: [Nat]) = CharTensor { charAsDynamic :: CharDynamic }
  deriving (Eq)
charAsStatic = CharTensor

type CCharStorage   = C'THCharStorage
newtype CharStorage = CharStorage { charStorageState :: (ForeignPtr CState, ForeignPtr CCharStorage) }
  deriving (Eq)
charStorage = curry CharStorage

-- Signed types

type CLongTensor      = C'THLongTensor
newtype LongDynamic = LongDynamic { longDynamicState :: (ForeignPtr CState, ForeignPtr CLongTensor) }
  deriving (Eq)
longDynamic = curry LongDynamic

newtype LongTensor (ds :: [Nat]) = LongTensor { longAsDynamic :: LongDynamic }
  deriving (Eq)
longAsStatic = LongTensor

type CLongStorage   = C'THLongStorage
newtype LongStorage = LongStorage { longStorageState :: (ForeignPtr CState, ForeignPtr CLongStorage) }
  deriving (Eq)
longStorage = curry LongStorage

type CShortTensor      = C'THShortTensor
newtype ShortDynamic = ShortDynamic { shortDynamicState :: (ForeignPtr CState, ForeignPtr CShortTensor) }
  deriving (Eq)
shortDynamic = curry ShortDynamic

newtype ShortTensor (ds :: [Nat]) = ShortTensor { shortAsDynamic :: ShortDynamic }
  deriving (Eq)
shortAsStatic = ShortTensor

type CShortStorage   = C'THShortStorage
newtype ShortStorage = ShortStorage { shortStorageState :: (ForeignPtr CState, ForeignPtr CShortStorage) }
  deriving (Eq)
shortStorage = curry ShortStorage

type CIntTensor      = C'THIntTensor
newtype IntDynamic = IntDynamic { intDynamicState :: (ForeignPtr CState, ForeignPtr CIntTensor) }
  deriving (Eq)
intDynamic = curry IntDynamic

newtype IntTensor (ds :: [Nat]) = IntTensor { intAsDynamic :: IntDynamic }
  deriving (Eq)
intAsStatic = IntTensor

type CIntStorage   = C'THIntStorage
newtype IntStorage = IntStorage { intStorageState :: (ForeignPtr CState, ForeignPtr CIntStorage) }
  deriving (Eq)
intStorage = curry IntStorage

-- Floating types

type CFloatTensor      = C'THFloatTensor
newtype FloatDynamic = FloatDynamic { floatDynamicState :: (ForeignPtr CState, ForeignPtr CFloatTensor) }
  deriving (Eq)
floatDynamic = curry FloatDynamic

newtype FloatTensor (ds :: [Nat]) = FloatTensor { floatAsDynamic :: FloatDynamic }
  deriving (Eq)
floatAsStatic = FloatTensor

type CFloatStorage   = C'THFloatStorage
newtype FloatStorage = FloatStorage { floatStorageState :: (ForeignPtr CState, ForeignPtr CFloatStorage) }
  deriving (Eq)
floatStorage = curry FloatStorage

type CDoubleTensor      = C'THDoubleTensor
newtype DoubleDynamic = DoubleDynamic { doubleDynamicState :: (ForeignPtr CState, ForeignPtr CDoubleTensor) }
  deriving (Eq)
doubleDynamic = curry DoubleDynamic

newtype DoubleTensor (ds :: [Nat]) = DoubleTensor { doubleAsDynamic :: DoubleDynamic }
  deriving (Eq)
doubleAsStatic = DoubleTensor

type CDoubleStorage   = C'THDoubleStorage
newtype DoubleStorage = DoubleStorage { doubleStorageState :: (ForeignPtr CState, ForeignPtr CDoubleStorage) }
  deriving (Eq)
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


