{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE UndecidableInstances #-}
module Torch.Core.Tensor.Types
  ( TensorFloat(..)
  , TensorDouble(..)
  , TensorByte(..)
  , TensorChar(..)
  , TensorShort(..)
  , TensorInt(..)
  , TensorLong(..)

  , TensorFloatRaw
  , TensorDoubleRaw
  , TensorByteRaw
  , TensorCharRaw
  , TensorShortRaw
  , TensorIntRaw
  , TensorLongRaw
  ) where


import Data.Functor.Identity
import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Data.Proxy

import Foreign.ForeignPtr (ForeignPtr, withForeignPtr, mallocForeignPtrArray, newForeignPtr)
import GHC.Ptr (FunPtr)

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
newtype TensorFloat = TensorFloat { tfTensor :: ForeignPtr CTHFloatTensor }
  deriving (Show, Eq)

newtype TensorDouble = TensorDouble { tdTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Eq, Show)


-- Int types
newtype TensorByte = TensorByte { tbTensor :: ForeignPtr CTHByteTensor }
  deriving (Eq, Show)

newtype TensorChar = TensorChar { tcTensor :: ForeignPtr CTHCharTensor }
  deriving (Eq, Show)

newtype TensorShort = TensorShort { tsTensor :: ForeignPtr CTHShortTensor }
  deriving (Eq, Show)

newtype TensorInt = TensorInt { tiTensor :: ForeignPtr CTHIntTensor }
  deriving (Eq, Show)

newtype TensorLong = TensorLong { tlTensor :: ForeignPtr CTHLongTensor }
  deriving (Eq, Show)
