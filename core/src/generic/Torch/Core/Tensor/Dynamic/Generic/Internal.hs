{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Core.Tensor.Dynamic.Generic.Internal
  ( THTensor(..)
  , TensorPtrType
  , GenericOps'
  , GenericMath'
  , HaskType'

  , TensorDouble(..)
  , TensorFloat(..)
  ) where

import Foreign (ForeignPtr)

import Torch.Core.Tensor.Generic.Internal (CTHDoubleTensor, CTHFloatTensor, HaskType)
import Torch.Core.Tensor.Generic (GenericOps, GenericMath)


class THTensor t where
  construct :: ForeignPtr (TensorPtrType t) -> t
  getForeign :: t -> ForeignPtr (TensorPtrType t)

type family TensorPtrType t
type instance TensorPtrType TensorDouble = CTHDoubleTensor
type instance TensorPtrType TensorFloat = CTHFloatTensor

newtype TensorDouble = TensorDouble { tdTensor :: ForeignPtr CTHDoubleTensor }
  deriving (Show, Eq)

instance THTensor TensorDouble where
  construct = TensorDouble
  getForeign = tdTensor

newtype TensorFloat = TensorFloat { tfTensor :: ForeignPtr CTHFloatTensor }
  deriving (Show, Eq)

instance THTensor TensorFloat where
  construct = TensorFloat
  getForeign = tfTensor

type GenericMath' t = (GenericOps (TensorPtrType t), GenericMath (TensorPtrType t), THTensor t)
type GenericOps' t = (GenericOps (TensorPtrType t), THTensor t)
type HaskType' t = (HaskType (TensorPtrType t))


