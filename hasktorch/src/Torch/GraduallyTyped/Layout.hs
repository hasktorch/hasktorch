{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Layout where

import Data.Kind (Type)
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen (kSparse, kStrided)
import qualified Torch.Internal.Type as ATen (Layout, Tensor, TensorList)

-- | Data type that represents the memory layout of a tensor.
data LayoutType
  = -- | dense (strided) tensor
    Dense
  | -- | sparse tensor
    Sparse

instance Castable LayoutType ATen.Layout where
  cast Dense f = f ATen.kStrided
  cast Sparse f = f ATen.kSparse

  uncast x f
    | x == ATen.kStrided = f Dense
    | x == ATen.kSparse = f Sparse

data Layout (layoutType :: Type) where
  AnyLayout :: forall layoutType. Layout layoutType
  Layout :: forall layoutType. layoutType -> Layout layoutType

class KnownLayout (layout :: Layout LayoutType) where
  layoutVal :: Layout LayoutType

instance KnownLayout 'AnyLayout where
  layoutVal = AnyLayout

instance KnownLayout ('Layout 'Dense) where
  layoutVal = Layout Dense

instance KnownLayout ('Layout 'Sparse) where
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
