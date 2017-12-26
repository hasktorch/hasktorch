{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE UndecidableInstances #-}
module Torch.Core.Tensor.Types (
  TensorFloat(..),
  TensorDouble(..),
  TensorByte(..),
  TensorChar(..),
  TensorShort(..),
  TensorInt(..),
  TensorLong(..),

  TensorFloatRaw,
  TensorDoubleRaw,
  TensorByteRaw,
  TensorCharRaw,
  TensorShortRaw,
  TensorIntRaw,
  TensorLongRaw,

  (^.), -- re-export for dimension tuple access
  _1, _2, _3, _4, _5
  ) where


import Data.Functor.Identity
import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Data.Proxy

import Foreign.ForeignPtr( ForeignPtr, withForeignPtr, mallocForeignPtrArray,
                           newForeignPtr )
import GHC.Ptr (FunPtr)

import Lens.Micro

import THTypes
import THDoubleTensor
import Numeric.Dimensions (Dim, SomeDims)

type TensorFloatRaw  = Ptr CTHFloatTensor
type TensorDoubleRaw = Ptr CTHDoubleTensor
type TensorByteRaw   = Ptr CTHByteTensor
type TensorCharRaw   = Ptr CTHCharTensor
type TensorShortRaw  = Ptr CTHShortTensor
type TensorIntRaw    = Ptr CTHIntTensor
type TensorLongRaw   = Ptr CTHLongTensor

-- Float types

data TensorFloat = TensorFloat {
  tfTensor :: !(ForeignPtr CTHFloatTensor),
  tfDim :: SomeDims
  } deriving (Show, Eq)

data TensorDouble = TensorDouble {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: SomeDims
  } deriving (Eq, Show)


-- Int types

data TensorByte = TensorByte {
  tbTensor :: !(ForeignPtr CTHByteTensor),
  tbDim :: SomeDims
  } deriving (Eq, Show)

data TensorChar = TensorChar {
  tcTensor :: !(ForeignPtr CTHCharTensor),
  tcDim :: SomeDims
  } deriving (Eq, Show)

data TensorShort = TensorShort {
  tsTensor :: !(ForeignPtr CTHShortTensor),
  tsDim :: SomeDims
  } deriving (Eq, Show)

data TensorInt = TensorInt {
  tiTensor :: !(ForeignPtr CTHIntTensor),
  tiDim :: SomeDims
  } deriving (Eq, Show)

data TensorLong = TensorLong {
  tlTensor :: !(ForeignPtr CTHLongTensor),
  tlDim :: SomeDims
  } deriving (Eq, Show)
