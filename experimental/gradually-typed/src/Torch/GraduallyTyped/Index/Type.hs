{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Torch.GraduallyTyped.Index.Type where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..))
import GHC.TypeNats (KnownNat, Nat, SomeNat (..), natVal, someNatVal)
import Numeric.Natural (Natural)
import Torch.GraduallyTyped.Prelude (IsChecked (..))

data Index (index :: Type) where
  UncheckedIndex :: forall index. Index index
  Index :: forall index. index -> Index index
  deriving (Show)

data SIndex (index :: Index Nat) where
  SUncheckedIndex :: Natural -> SIndex 'UncheckedIndex
  SIndex :: forall index. KnownNat index => SIndex ('Index index)

deriving stock instance Show (SIndex (index :: Index Nat))

type instance Sing = SIndex

instance KnownNat index => SingI ('Index index) where
  sing = SIndex

type family IndexF (index :: Index Nat) :: Nat where
  IndexF ('Index index) = index

instance SingKind (Index Nat) where
  type Demote (Index Nat) = IsChecked Natural
  fromSing (SUncheckedIndex index) = Unchecked index
  fromSing (SIndex :: SIndex index) = Checked . natVal $ Proxy @(IndexF index)
  toSing (Unchecked index) = SomeSing $ SUncheckedIndex index
  toSing (Checked index) = case someNatVal index of
    SomeNat (_ :: Proxy index) -> SomeSing (SIndex @index)

class KnownIndex (index :: Index Nat) where
  indexVal :: Index Natural

instance KnownIndex 'UncheckedIndex where
  indexVal = UncheckedIndex

instance KnownNat index => KnownIndex ('Index index) where
  indexVal = Index (natVal $ Proxy @index)
