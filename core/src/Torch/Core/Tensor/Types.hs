{-# LANGUAGE MultiParamTypeClasses, TypeSynonymInstances,  FunctionalDependencies #-}
{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}
{-  LANGUAGE TypeFamilies #-}
{-  LANGUAGE TypeOperators #-}
{-  LANGUAGE DataKinds #-}
{-  LANGUAGE PolyKinds #-}
{-  LANGUAGE KindSignatures #-}
{-  LANGUAGE FlexibleInstances #-}
{-  LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.Core.Tensor.Types (
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
import Numeric.Dimensions (Dim)

type TensorFloatRaw  = Ptr CTHFloatTensor
type TensorDoubleRaw = Ptr CTHDoubleTensor
type TensorByteRaw   = Ptr CTHByteTensor
type TensorCharRaw   = Ptr CTHCharTensor
type TensorShortRaw  = Ptr CTHShortTensor
type TensorIntRaw    = Ptr CTHIntTensor
type TensorLongRaw   = Ptr CTHLongTensor

-- Float types

data TensorFloat dims = TensorFloat {
  tfTensor :: !(ForeignPtr CTHFloatTensor),
  tfDim :: Dim dims
  } deriving (Show, Eq)

data TensorDouble dims = TensorDouble {
  tdTensor :: !(ForeignPtr CTHDoubleTensor),
  tdDim :: Dim dims
  } deriving (Eq, Show)

-- Int types

data TensorByte dims = TensorByte {
  tbTensor :: !(ForeignPtr CTHByteTensor),
  tbDim :: Dim dims
  } deriving (Eq, Show)

data TensorChar dims = TensorChar {
  tcTensor :: !(ForeignPtr CTHCharTensor),
  tcDim :: Dim dims
  } deriving (Eq, Show)

data TensorShort dims = TensorShort {
  tsTensor :: !(ForeignPtr CTHShortTensor),
  tsDim :: Dim dims
  } deriving (Eq, Show)

data TensorInt dims = TensorInt {
  tiTensor :: !(ForeignPtr CTHIntTensor),
  tiDim :: Dim dims
  } deriving (Eq, Show)

data TensorLong dims = TensorLong {
  tlTensor :: !(ForeignPtr CTHLongTensor),
  tlDim :: Dim dims
  } deriving (Eq, Show)
