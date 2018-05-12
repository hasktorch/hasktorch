{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Types.THC
  ( module Torch.Types.THC.Structs

  , CState, State(..), asState
  , CAllocator, Allocator(..)
  , CDescBuff, DescBuff, descBuff
  , CGenerator, Generator(..), generatorToRng

  -- for nn-packages
  , CNNState
  , CDim
  , CNNGenerator

  , CInt'
  , CMaskTensor, CIndexTensor, CIndexStorage, C'THCudaIndexTensor
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

  , C'THCHalfStorage, C'THCudaHalfTensor, C'THCFile, C'THCHalf
  ) where

import Foreign
import Foreign.C.Types
import GHC.TypeLits
import Data.Char (chr)

import Torch.Types.THC.Structs

-- nn-package
type CNNState = C'THCState
type CDim = CInt
type CNNGenerator = ()

type CAllocator = ()
type  Allocator = ()

type CDescBuff = C'THCDescBuff
type  DescBuff = String
descBuff :: Ptr CDescBuff -> IO DescBuff
descBuff p = (map (chr . fromIntegral) . c'THCDescBuff'str) <$> peek p

type CState = C'THCState
newtype State = State { asForeign :: ForeignPtr CState }
  deriving (Eq)
asState = State

type CGenerator = C'_Generator
newtype Generator = Generator { rng :: ForeignPtr CGenerator }
  deriving (Eq, Show)

generatorToRng :: ForeignPtr CGenerator -> Generator
generatorToRng = Generator

type CInt' = CLLong
type Int' = Integer

-- Some type alias'
type C'THCudaIndexTensor = CLongTensor
type CMaskTensor   = CByteTensor
type CIndexTensor  = CLongTensor
type CIndexStorage = CLongStorage

type  MaskDynamic   = ByteDynamic
type  MaskTensor    =  ByteTensor
type  IndexDynamic  =  LongDynamic
type  IndexTensor   =  LongTensor
type  IndexStorage = LongStorage

-- unsigned types

type CByteTensor      = C'THCudaByteTensor
newtype ByteDynamic   = ByteDynamic { byteDynamicState :: (ForeignPtr CState, ForeignPtr CByteTensor) }
  deriving (Eq)
byteDynamic = curry ByteDynamic

newtype ByteTensor (ds :: [Nat]) = ByteTensor { byteAsDynamic :: ByteDynamic }
  deriving (Eq)
byteAsStatic = ByteTensor

type CByteStorage   = C'THCByteStorage
newtype ByteStorage = ByteStorage { byteStorageState :: (ForeignPtr CState, ForeignPtr CByteStorage) }
  deriving (Eq)
byteStorage = curry ByteStorage

type CCharTensor      = C'THCudaCharTensor
newtype CharDynamic = CharDynamic { charDynamicState :: (ForeignPtr CState, ForeignPtr CCharTensor) }
  deriving (Eq)
charDynamic = curry CharDynamic

newtype CharTensor (ds :: [Nat]) = CharTensor { charAsDynamic :: CharDynamic }
  deriving (Eq)
charAsStatic = CharTensor

type CCharStorage   = C'THCCharStorage
newtype CharStorage = CharStorage { charStorageState :: (ForeignPtr CState, ForeignPtr CCharStorage) }
  deriving (Eq)
charStorage = curry CharStorage

-- Signed types

type CLongTensor      = C'THCudaLongTensor
newtype LongDynamic = LongDynamic { longDynamicState :: (ForeignPtr CState, ForeignPtr CLongTensor) }
  deriving (Eq)
longDynamic = curry LongDynamic

newtype LongTensor (ds :: [Nat]) = LongTensor { longAsDynamic :: LongDynamic }
  deriving (Eq)
longAsStatic = LongTensor

type CLongStorage   = C'THCLongStorage
newtype LongStorage = LongStorage { longStorageState :: (ForeignPtr CState, ForeignPtr CLongStorage) }
  deriving (Eq)
longStorage = curry LongStorage

type CShortTensor      = C'THCudaShortTensor
newtype ShortDynamic = ShortDynamic { shortDynamicState :: (ForeignPtr CState, ForeignPtr CShortTensor) }
  deriving (Eq)
shortDynamic = curry ShortDynamic

newtype ShortTensor (ds :: [Nat]) = ShortTensor { shortAsDynamic :: ShortDynamic }
  deriving (Eq)
shortAsStatic = ShortTensor

type CShortStorage   = C'THCShortStorage
newtype ShortStorage = ShortStorage { shortStorageState :: (ForeignPtr CState, ForeignPtr CShortStorage) }
  deriving (Eq)
shortStorage = curry ShortStorage

type CIntTensor      = C'THCudaIntTensor
newtype IntDynamic = IntDynamic { intDynamicState :: (ForeignPtr CState, ForeignPtr CIntTensor) }
  deriving (Eq)
intDynamic = curry IntDynamic

newtype IntTensor (ds :: [Nat]) = IntTensor { intAsDynamic :: IntDynamic }
  deriving (Eq)
intAsStatic = IntTensor

type CIntStorage   = C'THCIntStorage
newtype IntStorage = IntStorage { intStorageState :: (ForeignPtr CState, ForeignPtr CIntStorage) }
  deriving (Eq)
intStorage = curry IntStorage

-- Floating types

type CFloatTensor      = C'THCudaFloatTensor
newtype FloatDynamic = FloatDynamic { floatDynamicState :: (ForeignPtr CState, ForeignPtr CFloatTensor) }
  deriving (Eq)
floatDynamic = curry FloatDynamic

newtype FloatTensor (ds :: [Nat]) = FloatTensor { floatAsDynamic :: FloatDynamic }
  deriving (Eq)
floatAsStatic = FloatTensor

type CFloatStorage   = C'THCFloatStorage
newtype FloatStorage = FloatStorage { floatStorageState :: (ForeignPtr CState, ForeignPtr CFloatStorage) }
  deriving (Eq)
floatStorage = curry FloatStorage

type CDoubleTensor      = C'THCudaDoubleTensor
newtype DoubleDynamic = DoubleDynamic { doubleDynamicState :: (ForeignPtr CState, ForeignPtr CDoubleTensor) }
  deriving (Eq)
doubleDynamic = curry DoubleDynamic

newtype DoubleTensor (ds :: [Nat]) = DoubleTensor { doubleAsDynamic :: DoubleDynamic }
  deriving (Eq)
doubleAsStatic = DoubleTensor

type CDoubleStorage   = C'THCDoubleStorage
newtype DoubleStorage = DoubleStorage { doubleStorageState :: (ForeignPtr CState, ForeignPtr CDoubleStorage) }
  deriving (Eq)
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


