{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module TensorTypes (
  TensorDim(..),
  TensorFloat(..),
  TensorDouble(..),
  TensorByte(..),
  TensorChar(..),
  TensorShort(..),
  TensorInt(..),
  TensorLong(..),

  TensorFloatRaw(..),
  TensorDoubleRaw(..),
  TensorByteRaw(..),
  TensorCharRaw(..),
  TensorShortRaw(..),
  TensorIntRaw(..),
  TensorLongRaw(..),

  TIdx(..),
  (^.), -- re-export for dimension tuple access
  _1, _2, _3, _4, _5
  ) where


import Data.Functor.Identity
import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import Lens.Micro
import Lens.Micro.Internal (Field1(..), Field2(..), Field3(..), Field4(..))
import Lens.Micro (lens)

import THTypes
import THDoubleTensor

type TensorFloatRaw  = Ptr CTHFloatTensor
type TensorDoubleRaw = Ptr CTHDoubleTensor
type TensorByteRaw   = Ptr CTHByteTensor
type TensorCharRaw   = Ptr CTHCharTensor
type TensorShortRaw  = Ptr CTHShortTensor
type TensorIntRaw    = Ptr CTHIntTensor
type TensorLongRaw   = Ptr CTHLongTensor

data TensorDim a =
  D0
  | D1 { d1 :: a }
  | D2 { d2 :: (a, a) }
  | D3 { d3 :: (a, a, a) }
  | D4 { d4 :: (a, a, a, a) }
  deriving (Eq, Show)

instance Functor TensorDim where
  fmap f D0 = D0
  fmap f (D1 d1) = D1 (f d1)
  fmap f (D2 (d1, d2)) = D2 ((f d1), (f d2))
  fmap f (D3 (d1, d2, d3)) = D3 ((f d1), (f d2), (f d3))
  fmap f (D4 (d1, d2, d3, d4)) = D4 ((f d1), (f d2), (f d3), (f d4))

instance Foldable TensorDim where
  foldr func val (D0) = val
  foldr func val (D1 d1) = foldr func val [d1]
  foldr func val (D2 (d1, d2)) = foldr func val [d1, d2]
  foldr func val (D3 (d1, d2, d3)) = foldr func val [d1, d2, d3]
  foldr func val (D4 (d1, d2, d3, d4)) = foldr func val [d1, d2, d3, d4]

-- Float types

data TensorFloat = TensorFloat {
  tfTensor :: !(ForeignPtr CTHFloatTensor),
  tfDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorDouble = TensorDouble {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: !(TensorDim Word)
  } deriving (Eq, Show)

-- Int types

data TensorByte = TensorByte {
  tbTensor :: !(ForeignPtr CTHByteTensor),
  tbDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorChar = TensorChar {
  tcTensor :: !(ForeignPtr CTHCharTensor),
  tcDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorShort = TensorShort {
  tsTensor :: !(ForeignPtr CTHShortTensor),
  tsDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorInt = TensorInt {
  tiTensor :: !(ForeignPtr CTHIntTensor),
  tiDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorLong = TensorLong {
  tlTensor :: !(ForeignPtr CTHLongTensor),
  tlDim :: !(TensorDim Word)
  } deriving (Eq, Show)

-- Indexing types:
data TIdx a where
  I0 :: TIdx a
  I1 :: a -> TIdx a
  I2 :: a -> a -> TIdx a
  I3 :: a -> a -> a -> TIdx a
  I4 :: a -> a -> a -> a -> TIdx a

instance Show a => Show (TIdx a) where
  show = \case
    I0 -> "[]"
    I1 a -> show [a]
    I2 a b -> show [a,b]
    I3 a b c -> show [a,b,c]
    I4 a b c d -> show [a,b,c,d]

_1_ :: TIdx a -> a
_1_ = \case
  I1 a -> a
  I2 a _ -> a
  I3 a _ _ -> a
  I4 a _ _ _ -> a
  _ -> error "no _1 index exists for this dimensionality"

_1'_ :: a -> TIdx a -> TIdx a
_1'_ a = \case
  I1 _ -> I1 a
  I2 _ b -> I2 a b
  I3 _ b c -> I3 a b c
  I4 _ b c d -> I4 a b c d
  x -> x

_2_ :: TIdx a -> a
_2_ = \case
  I2 _ a -> a
  I3 _ a _ -> a
  I4 _ a _ _ -> a
  _ -> error "no _2 index exists for this dimensionality"

_2'_ :: a -> TIdx a -> TIdx a
_2'_ b = \case
  I2 a _ -> I2 a b
  I3 a _ c -> I3 a b c
  I4 a _ c d -> I4 a b c d
  x -> x

_3_ :: TIdx a -> a
_3_ = \case
  I3 _ _ a -> a
  I4 _ _ a _ -> a
  _ -> error "no _3 index exists for this dimensionality"

_3'_ :: a -> TIdx a -> TIdx a
_3'_ c = \case
  I3 a b _ -> I3 a b c
  I4 a b _ d -> I4 a b c d
  x -> x

_4_ :: TIdx a -> a
_4_ = \case
  I4 _ _ _ a -> a
  _ -> error "no _4 index exists for this dimensionality"

_4'_ :: a -> TIdx a -> TIdx a
_4'_ d = \case
  I4 a b c _ -> I4 a b c d
  x -> x

instance Field1 (TIdx a) (TIdx a) a a where
  _1 = lens _1_ (flip _1'_)
instance Field2 (TIdx a) (TIdx a) a a where
  _2 = lens _2_ (flip _2'_)
instance Field3 (TIdx a) (TIdx a) a a where
  _3 = lens _3_ (flip _3'_)
instance Field4 (TIdx a) (TIdx a) a a where
  _4 = lens _4_ (flip _4'_)


