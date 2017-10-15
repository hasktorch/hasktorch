module TensorTypes (
  TensorDim(..),
  TensorFloat(..),
  TensorDouble_(..),
  TensorByte(..),
  TensorChar(..),
  TensorShort(..),
  TensorInt(..),
  TensorLong(..)
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import THTypes
import THDoubleTensor

-- TODO what should happen when accessor are applied to the wrong type here?
data TensorDim a =
  D0
  | D1 { d1_1 :: a }
  | D2 { d2_1 :: a, d2_2 :: a }
  | D3 { d3_1 :: a, d3_2 :: a, d3_3 :: a }
  | D4 { d4_1 :: a, d4_2 :: a, d4_3 :: a, d4_4 :: a }
  deriving (Eq, Show)

instance Functor TensorDim where
  fmap f D0 = D0
  fmap f (D1 d1) = D1 (f d1)
  fmap f (D2 d1 d2) = D2 (f d1) (f d2)
  fmap f (D3 d1 d2 d3) = D3 (f d1) (f d2) (f d3)
  fmap f (D4 d1 d2 d3 d4) = D4 (f d1) (f d2) (f d3) (f d4)

instance Foldable TensorDim where
  foldr func val (D0) = val
  foldr func val (D1 d1) = foldr func val [d1]
  foldr func val (D2 d1 d2) = foldr func val [d1, d2]
  foldr func val (D3 d1 d2 d3) = foldr func val [d1, d2, d3]
  foldr func val (D4 d1 d2 d3 d4) = foldr func val [d1, d2, d3, d4]

-- Float types

data TensorFloat = TensorFloat {
  tfTensor :: !(ForeignPtr CTHFloatTensor),
  tfDim :: !(TensorDim Word)
  } deriving (Eq, Show)

data TensorDouble_ = TensorDouble_ {
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
