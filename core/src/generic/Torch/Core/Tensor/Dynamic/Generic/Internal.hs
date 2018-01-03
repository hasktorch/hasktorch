{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Dynamic.Generic.Internal
  ( THTensor(..)
  , TensorPtrType
  , GenericOps'
  , GenericMath'
  , HaskReal'

  , TensorByte(..)
  , TensorDouble(..)
  , TensorFloat(..)
  ) where

import Foreign (ForeignPtr)

import Torch.Core.Tensor.Generic.Internal
import Torch.Core.Tensor.Generic (GenericOps, GenericMath)

type GenericMath' t = (GenericOps (TensorPtrType t), GenericMath (TensorPtrType t), THTensor t)
type GenericOps' t = (GenericOps (TensorPtrType t), THTensor t)
type HaskReal' t = (HaskReal (TensorPtrType t))


class THTensor t where
  construct :: ForeignPtr (TensorPtrType t) -> t
  getForeign :: t -> ForeignPtr (TensorPtrType t)

type family TensorPtrType t
type instance TensorPtrType TensorByte = CTHByteTensor
type instance TensorPtrType TensorDouble = CTHDoubleTensor
type instance TensorPtrType TensorFloat = CTHFloatTensor
type instance TensorPtrType TensorInt = CTHIntTensor
type instance TensorPtrType TensorLong = CTHLongTensor
type instance TensorPtrType TensorShort = CTHShortTensor

-------------------------------------------------------------------------------

newtype TensorByte = TensorByte { tbTensor :: ForeignPtr CTHByteTensor }
  deriving (Show, Eq)

instance THTensor TensorByte where
  construct = TensorByte
  getForeign = tbTensor

-------------------------------------------------------------------------------

newtype TensorDouble = TensorDouble { tdTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Show, Eq)

instance THTensor TensorDouble where
  construct = TensorDouble
  getForeign = tdTensor

-------------------------------------------------------------------------------

newtype TensorFloat = TensorFloat { tfTensor :: ForeignPtr CTHFloatTensor }
  deriving (Show, Eq)

instance THTensor TensorFloat where
  construct = TensorFloat
  getForeign = tfTensor

-------------------------------------------------------------------------------

newtype TensorInt = TensorInt { tiTensor :: ForeignPtr CTHIntTensor }
  deriving (Show, Eq)

instance THTensor TensorInt where
  construct = TensorInt
  getForeign = tiTensor

-------------------------------------------------------------------------------

newtype TensorLong = TensorLong { tlTensor :: ForeignPtr CTHLongTensor }
  deriving (Show, Eq)

instance THTensor TensorLong where
  construct = TensorLong
  getForeign = tlTensor

-------------------------------------------------------------------------------

newtype TensorShort = TensorShort { tsTensor :: ForeignPtr CTHShortTensor }
  deriving (Show, Eq)

instance THTensor TensorShort where
  construct = TensorShort
  getForeign = tsTensor


