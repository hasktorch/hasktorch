{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.Layout where

import Data.Kind (Type)
import Data.Singletons (Sing)
import Data.Singletons.Prelude.Check (Check, SCheck (..), type SChecked, type SUnchecked)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat)
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

instance Castable LayoutType ATen.Layout where
  cast Dense f = f ATen.kStrided
  cast Sparse f = f ATen.kSparse

  uncast x f
    | x == ATen.kStrided = f Dense
    | x == ATen.kSparse = f Sparse

type SLayout :: LayoutType -> Type

type SLayout layoutType = SChecked layoutType

pattern SLayout :: forall (a :: LayoutType). Sing a -> SLayout a
pattern SLayout layoutType = SChecked layoutType

type SUncheckedLayout :: Type

type SUncheckedLayout = SUnchecked LayoutType

pattern SUncheckedLayout :: LayoutType -> SUncheckedLayout
pattern SUncheckedLayout layoutType = SUnchecked layoutType

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
type GetLayouts :: k -> [Check LayoutType Type]
type family GetLayouts f where
  GetLayouts (a :: Check LayoutType Type) = '[a]
  GetLayouts (f g) = Concat (GetLayouts f) (GetLayouts g)
  GetLayouts _ = '[]
