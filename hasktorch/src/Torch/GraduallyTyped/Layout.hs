{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Layout where

import Data.Kind (Type)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen (kSparse, kStrided)
import qualified Torch.Internal.Type as ATen (Layout)
import Type.Errors.Pretty (TypeError, type (%), type (<>))

-- | Data type that represents the memory layout of a tensor.
data LayoutType
  = -- | The memory layout of the tensor is dense (strided).
    Dense
  | -- | The memory layout of the tensor is sparse.
    Sparse
  deriving (Show, Eq)

instance Castable LayoutType ATen.Layout where
  cast Dense f = f ATen.kStrided
  cast Sparse f = f ATen.kSparse

  uncast x f
    | x == ATen.kStrided = f Dense
    | x == ATen.kSparse = f Sparse

-- | Data type to represent whether or not the tensor's memory layout is checked, that is, known to the compiler.
data Layout (layoutType :: Type) where
  -- | The tensor's memory layout is unknown to the compiler.
  UncheckedLayout :: forall layoutType. Layout layoutType
  -- | The tensor's memory layout is known to the compiler.
  Layout :: forall layoutType. layoutType -> Layout layoutType
  deriving (Show)

class KnownLayout (layout :: Layout LayoutType) where
  layoutVal :: Layout LayoutType

instance KnownLayout 'UncheckedLayout where
  layoutVal = UncheckedLayout

instance KnownLayout ( 'Layout 'Dense) where
  layoutVal = Layout Dense

instance KnownLayout ( 'Layout 'Sparse) where
  layoutVal = Layout Sparse

class WithLayoutC (isAnyLayout :: Bool) (layout :: Layout LayoutType) (f :: Type) where
  type WithLayoutF isAnyLayout f :: Type
  withLayout :: (LayoutType -> f) -> WithLayoutF isAnyLayout f

instance WithLayoutC 'True layout f where
  type WithLayoutF 'True f = LayoutType -> f
  withLayout = id

instance (KnownLayout layout) => WithLayoutC 'False layout f where
  type WithLayoutF 'False f = f
  withLayout f = case layoutVal @layout of Layout layout -> f layout

type family UnifyLayoutF (layout :: Layout LayoutType) (layout' :: Layout LayoutType) :: Layout LayoutType where
  UnifyLayoutF 'UncheckedLayout 'UncheckedLayout = 'UncheckedLayout
  UnifyLayoutF ( 'Layout _) 'UncheckedLayout = 'UncheckedLayout
  UnifyLayoutF 'UncheckedLayout ( 'Layout _) = 'UncheckedLayout
  UnifyLayoutF ( 'Layout layoutType) ( 'Layout layoutType) = 'Layout layoutType
  UnifyLayoutF ( 'Layout layoutType) ( 'Layout layoutType') =
    TypeError
      ( "The supplied tensors must have the same memory layout,"
          % "but different layouts were found:"
          % ""
          % "    " <> layoutType <> " and " <> layoutType' <> "."
          % ""
      )