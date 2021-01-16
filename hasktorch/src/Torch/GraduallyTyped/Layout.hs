{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
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
import Type.Errors.Pretty (type (%), type (<>))

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
