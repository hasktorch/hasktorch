{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE NoStarIsType #-}

module Data.Singletons.Prelude.Check where

import Data.Kind (Type)
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)

type Check :: Type -> Type -> Type
data Check a b = Checked a | Unchecked b

forgetIsChecked :: Check a a -> a
forgetIsChecked (Checked a) = a
forgetIsChecked (Unchecked a) = a

type SChecked :: k -> Type

type SChecked (a :: k) = SCheck ('Checked @k @Type a)

type SUnchecked :: Type -> Type

type SUnchecked b = SCheck ('Unchecked @b @Type b)

type SCheck :: Check a Type -> Type
data SCheck checked where
  SUnchecked :: forall b. b -> SUnchecked b
  SChecked :: forall a. Sing a -> SChecked a

type instance Sing = SCheck

instance SingKind a => SingKind (Check a Type) where
  type Demote (Check a Type) = Check (Demote a) a
  fromSing (SUnchecked a) = Unchecked a
  fromSing (SChecked a) = Checked (fromSing a)
  toSing (Unchecked a) = SomeSing (SUnchecked a)
  toSing (Checked a) = withSomeSing a $ SomeSing . SChecked

instance SingI (a :: k) => SingI ('Checked @k @Type a) where
  sing = SChecked $ sing @a

pattern Demoted :: forall a (checked :: Check a Type). (SingKind a, Demote a ~ a) => a -> SCheck checked
pattern Demoted unchecked <- (forgetIsChecked . fromSing -> unchecked)

{-# COMPLETE Demoted #-}
