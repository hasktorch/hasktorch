{-# LANGUAGE ConstraintKinds #-}
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

import Data.Kind (Constraint, Type)
import Torch.GraduallyTyped.Prelude (Catch)
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

class KnownLayoutType (layoutType :: LayoutType) where
  layoutTypeVal :: LayoutType

instance KnownLayoutType 'Dense where
  layoutTypeVal = Dense

instance KnownLayoutType 'Sparse where
  layoutTypeVal = Sparse

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

instance (KnownLayoutType layoutType) => KnownLayout ( 'Layout layoutType) where
  layoutVal = Layout (layoutTypeVal @layoutType)

class WithLayoutC (layout :: Layout LayoutType) (f :: Type) where
  type WithLayoutF layout f :: Type
  withLayout :: (LayoutType -> f) -> WithLayoutF layout f
  withoutLayout :: WithLayoutF layout f -> (LayoutType -> f)

instance WithLayoutC 'UncheckedLayout f where
  type WithLayoutF 'UncheckedLayout f = LayoutType -> f
  withLayout = id
  withoutLayout = id

instance (KnownLayoutType layoutType) => WithLayoutC ( 'Layout layoutType) f where
  type WithLayoutF ( 'Layout layoutType) f = f
  withLayout f = f (layoutTypeVal @layoutType)
  withoutLayout = const

type UnifyLayoutRightAssociativeL layout layout' layout'' = UnifyLayoutF (UnifyLayoutF layout layout') layout'' ~ UnifyLayoutF layout (UnifyLayoutF layout' layout'')
type UnifyLayoutIdempotenceL1 layout = UnifyLayoutF layout layout ~ layout
type UnifyLayoutIdempotenceL2 layout layout' = UnifyLayoutF layout (UnifyLayoutF layout layout') ~ UnifyLayoutF layout layout'
type UnifyLayoutIdempotenceL2C layout layout' = UnifyLayoutF layout (UnifyLayoutF layout' layout) ~ UnifyLayoutF layout layout'
type UnifyLayoutIdempotenceL3 layout layout' layout'' = UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout layout'')) ~ UnifyLayoutF layout (UnifyLayoutF layout' layout'')
type UnifyLayoutIdempotenceL3C layout layout' layout'' = UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout'' layout)) ~ UnifyLayoutF layout (UnifyLayoutF layout' layout'')
type UnifyLayoutIdempotenceL4 layout layout' layout'' layout''' = UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout'' (UnifyLayoutF layout layout'''))) ~ UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout'' layout'''))
type UnifyLayoutIdempotenceL4C layout layout' layout'' layout''' = UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout'' (UnifyLayoutF layout''' layout))) ~ UnifyLayoutF layout (UnifyLayoutF layout' (UnifyLayoutF layout'' layout'''))

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

type family UnifyLayoutC (layout :: Layout LayoutType) (layout' :: Layout LayoutType) :: Constraint where
  UnifyLayoutC layout layout' = Catch (UnifyLayoutF layout layout')