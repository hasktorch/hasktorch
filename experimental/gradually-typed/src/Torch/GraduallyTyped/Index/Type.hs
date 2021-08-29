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
import Data.Maybe (fromJust)
import Data.Proxy (Proxy (..))
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..))
import GHC.TypeLits (KnownNat, Nat, SomeNat (..), natVal, someNatVal)
import Torch.GraduallyTyped.Prelude (IsChecked (..))

data Index (index :: Type) where
  UncheckedIndex :: forall index. Index index
  Index :: forall index. index -> Index index
  NegativeIndex :: forall index. index -> Index index
  deriving (Show)

data SIndex (index :: Index Nat) where
  SUncheckedIndex :: Integer -> SIndex 'UncheckedIndex
  SIndex :: forall index. KnownNat index => SIndex ('Index index)
  SNegativeIndex :: forall index. KnownNat index => SIndex ('NegativeIndex index)

deriving stock instance Show (SIndex (index :: Index Nat))

type instance Sing = SIndex

instance KnownNat index => SingI ('Index index) where
  sing = SIndex

instance KnownNat index => SingI ('NegativeIndex index) where
  sing = SNegativeIndex

type family IndexF (index :: Index Nat) :: Nat where
  IndexF ('Index index) = index
  IndexF ('NegativeIndex index) = index

newtype DemotedIndex = DemotedIndex Integer

instance SingKind (Index Nat) where
  type Demote (Index Nat) = IsChecked DemotedIndex
  fromSing (SUncheckedIndex index) = Unchecked $ DemotedIndex index
  fromSing (SIndex :: SIndex index) = Checked . DemotedIndex . natVal $ Proxy @(IndexF index)
  fromSing (SNegativeIndex :: SIndex index) = Checked . DemotedIndex . negate . natVal $ Proxy @(IndexF index)
  toSing (Unchecked (DemotedIndex index)) = SomeSing $ SUncheckedIndex index
  toSing (Checked (DemotedIndex index)) =
    if index < 0
      then case fromJust $ someNatVal $ negate index of
        SomeNat (_ :: Proxy index) -> SomeSing (SNegativeIndex @index)
      else case fromJust $ someNatVal index of
        SomeNat (_ :: Proxy index) -> SomeSing (SIndex @index)
