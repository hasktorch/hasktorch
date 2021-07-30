{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.Layout where

import Data.Kind (Type)
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..))
import Torch.Internal.Class (Castable (..))
import qualified Torch.Internal.Const as ATen (kSparse, kStrided)
import qualified Torch.Internal.Type as ATen (Layout)

-- | Data type that represents the memory layout of a tensor.
data LayoutType
  = -- | The memory layout of the tensor is dense (strided).
    Dense
  | -- | The memory layout of the tensor is sparse.
    Sparse
  deriving (Show, Eq)

genSingletons [''LayoutType]

deriving stock instance Show (SLayoutType (layoutType :: LayoutType))

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

data SLayout (layout :: Layout LayoutType) where
  SUncheckedLayout :: LayoutType -> SLayout 'UncheckedLayout
  SLayout :: forall layoutType. SLayoutType layoutType -> SLayout ('Layout layoutType)

deriving stock instance Show (SLayout (layout :: Layout LayoutType))

type instance Sing = SLayout

instance SingI layoutType => SingI ('Layout (layoutType :: LayoutType)) where
  sing = SLayout $ sing @layoutType

instance SingKind (Layout LayoutType) where
  type Demote (Layout LayoutType) = IsChecked LayoutType
  fromSing (SUncheckedLayout layoutType) = Unchecked layoutType
  fromSing (SLayout layoutType) = Checked . fromSing $ layoutType
  toSing (Unchecked layoutType) = SomeSing (SUncheckedLayout layoutType)
  toSing (Checked layoutType) = withSomeSing layoutType $ SomeSing . SLayout

class KnownLayout (layout :: Layout LayoutType) where
  layoutVal :: Layout LayoutType

instance KnownLayout 'UncheckedLayout where
  layoutVal = UncheckedLayout

instance (KnownLayoutType layoutType) => KnownLayout ('Layout layoutType) where
  layoutVal = Layout (layoutTypeVal @layoutType)

-- >>> :kind! GetLayouts ('Layout 'Dense)
-- GetLayouts ('Layout 'Dense) :: [Layout LayoutType]
-- = '[ 'Layout 'Dense]
-- >>> :kind! GetLayouts '[ 'Layout 'Sparse, 'Layout 'Dense]
-- GetLayouts '[ 'Layout 'Sparse, 'Layout 'Dense] :: [Layout
--                                                      LayoutType]
-- = '[ 'Layout 'Sparse, 'Layout 'Dense]
-- >>> :kind! GetLayouts ('Just ('Layout 'Dense))
-- GetLayouts ('Just ('Layout 'Dense)) :: [Layout LayoutType]
-- = '[ 'Layout 'Dense]
type GetLayouts :: k -> [Layout LayoutType]
type family GetLayouts f where
  GetLayouts (a :: Layout LayoutType) = '[a]
  GetLayouts (f g) = Concat (GetLayouts f) (GetLayouts g)
  GetLayouts _ = '[]
