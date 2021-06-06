{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.NamedTensor where

import Data.Default.Class
import Data.Kind
import Data.Maybe (fromJust)
import Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as V
import GHC.Exts
import GHC.Generics
import GHC.TypeLits
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Lens
import qualified Torch.Tensor as D
import Torch.Typed.Factories
import Torch.Typed.Functional
import Torch.Typed.Tensor

class NamedTensorLike a where
  type ToNestedList a :: Type
  toNestedList :: a -> ToNestedList a
  asNamedTensor :: a -> NamedTensor '( 'D.CPU, 0) (ToDType a) (ToShape a)
  fromNestedList :: ToNestedList a -> a
  fromNamedTensor :: NamedTensor '( 'D.CPU, 0) (ToDType a) (ToShape a) -> a

instance NamedTensorLike Bool where
  type ToNestedList Bool = Bool
  toNestedList = id
  asNamedTensor = fromUnnamed . UnsafeMkTensor . D.asTensor
  fromNestedList = id
  fromNamedTensor = D.asValue . toDynamic

instance NamedTensorLike Int where
  type ToNestedList Int = Int
  toNestedList = id
  asNamedTensor = fromUnnamed . UnsafeMkTensor . D.asTensor
  fromNestedList = id
  fromNamedTensor = D.asValue . toDynamic

instance NamedTensorLike Float where
  type ToNestedList Float = Float
  toNestedList = id
  asNamedTensor = fromUnnamed . UnsafeMkTensor . D.asTensor
  fromNestedList = id
  fromNamedTensor = D.asValue . toDynamic

instance NamedTensorLike Double where
  type ToNestedList Double = Double
  toNestedList = id
  asNamedTensor = fromUnnamed . UnsafeMkTensor . D.asTensor
  fromNestedList = id
  fromNamedTensor = D.asValue . toDynamic

instance (KnownNat n, D.TensorLike (ToNestedList a), NamedTensorLike a) => NamedTensorLike (Vector n a) where
  type ToNestedList (Vector n a) = [ToNestedList a]
  toNestedList v = fmap toNestedList (V.toList v)
  asNamedTensor v = fromUnnamed . UnsafeMkTensor . D.asTensor $ toNestedList v
  fromNestedList = fmap fromNestedList . fromJust . V.fromList
  fromNamedTensor = fromNestedList . D.asValue . toDynamic

instance {-# OVERLAPS #-} (Coercible (vec n a) (Vector n a), KnownNat n, D.TensorLike (ToNestedList a), NamedTensorLike a) => NamedTensorLike (vec n a) where
  type ToNestedList (vec n a) = [ToNestedList a]
  toNestedList v = map (toNestedList @a) (V.toList (coerce v :: Vector n a))
  asNamedTensor v = fromUnnamed . UnsafeMkTensor . D.asTensor $ toNestedList v
  fromNestedList v = coerce (fmap fromNestedList . fromJust . V.fromList $ v :: Vector n a)
  fromNamedTensor = fromNestedList . D.asValue . toDynamic

instance {-# OVERLAPS #-} (Generic (g a), Default (g a), HasTypes (g a) a, KnownNat (ToNat g), D.TensorLike (ToNestedList a), NamedTensorLike a) => NamedTensorLike (g a) where
  type ToNestedList (g a) = [ToNestedList a]
  toNestedList v = map (toNestedList @a) (flattenValues (types @a) v)
  asNamedTensor v = fromUnnamed . UnsafeMkTensor . D.asTensor $ toNestedList v
  fromNestedList v = replaceValues (types @a) def (fmap fromNestedList v)
  fromNamedTensor = fromNestedList . D.asValue . toDynamic
