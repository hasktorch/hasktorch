{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.GraduallyTyped.Index.Type where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, Symbol, natVal)
import Torch.GraduallyTyped.Shape (By, SelectDim (..), Shape, Dim(..), Name (..), Size (..))

data Index (index :: Type) where
  UncheckedIndex :: forall index. Index index
  Index :: forall index. index -> Index index
  deriving (Show)

class KnownIndex (index :: Index Nat) where
  indexVal :: Index Integer

instance KnownIndex 'UncheckedIndex where
  indexVal = UncheckedIndex

instance KnownNat index => KnownIndex ('Index index) where
  indexVal = Index (natVal $ Proxy @index)

class WithIndexC (index :: Index Nat) (f :: Type) where
  type WithIndexF index f :: Type
  withIndex :: (Integer -> f) -> WithIndexF index f
  withoutIndex :: WithIndexF index f -> (Integer -> f)

instance WithIndexC 'UncheckedIndex f where
  type WithIndexF 'UncheckedIndex f = Integer -> f
  withIndex = id
  withoutIndex = id

instance (KnownNat index) => WithIndexC ('Index index) f where
  type WithIndexF ('Index index) f = f
  withIndex f = f . natVal $ Proxy @index
  withoutIndex = const
