{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.GraduallyTyped.RequiresGradient where

import Data.Kind (Type)
import Data.Singletons (Sing, SingI (..), SingKind (..), SomeSing (..), withSomeSing)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat, IsChecked (..))

-- | Data type to represent whether or not the tensor requires gradient computations.
data RequiresGradient
  = -- | The tensor requires gradient computations.
    WithGradient
  | -- | Gradient computations for this tensor are disabled.
    WithoutGradient
  deriving (Show, Eq)

genSingletons [''RequiresGradient]

deriving stock instance Show (SRequiresGradient (requiresGradient :: RequiresGradient))

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'WithGradient where
  requiresGradientVal = WithGradient

instance KnownRequiresGradient 'WithoutGradient where
  requiresGradientVal = WithoutGradient

-- | Data type to represent whether or not it is known by the compiler if the tensor requires gradient computations.
data Gradient (requiresGradient :: Type) where
  -- | Whether or not the tensor requires gradient computations is unknown to the compiler.
  UncheckedGradient :: forall requiresGradient. Gradient requiresGradient
  -- | Whether or not the tensor requires gradient computations is known to the compiler.
  Gradient :: forall requiresGradient. requiresGradient -> Gradient requiresGradient
  deriving (Show)

data SGradient (gradient :: Gradient RequiresGradient) where
  SUncheckedGradient :: RequiresGradient -> SGradient 'UncheckedGradient
  SGradient :: forall requiresGradient. SRequiresGradient requiresGradient -> SGradient ('Gradient requiresGradient)

deriving stock instance Show (SGradient (requiresGradient :: Gradient RequiresGradient))

type instance Sing = SGradient

instance SingI requiresGradient => SingI ('Gradient (requiresGradient :: RequiresGradient)) where
  sing = SGradient $ sing @requiresGradient

instance SingKind (Gradient RequiresGradient) where
  type Demote (Gradient RequiresGradient) = IsChecked RequiresGradient
  fromSing (SUncheckedGradient requiresGradient) = Unchecked requiresGradient
  fromSing (SGradient requiresGradient) = Checked . fromSing $ requiresGradient
  toSing (Unchecked requiresGradient) = SomeSing (SUncheckedGradient requiresGradient)
  toSing (Checked requiresGradient) = withSomeSing requiresGradient $ SomeSing . SGradient

class KnownGradient (gradient :: Gradient RequiresGradient) where
  gradientVal :: Gradient RequiresGradient

instance KnownGradient 'UncheckedGradient where
  gradientVal = UncheckedGradient

instance (KnownRequiresGradient requiresGradient) => KnownGradient ('Gradient requiresGradient) where
  gradientVal = Gradient (requiresGradientVal @requiresGradient)

-- >>> :kind! GetGradients ('Gradient 'WithGradient)
-- GetGradients ('Gradient 'WithGradient) :: [Gradient RequiresGradient]
-- = '[ 'Gradient 'WithGradient]
-- >>> :kind! GetGradients '[ 'Gradient 'WithoutGradient, 'Gradient 'WithGradient]
-- GetGradients '[ 'Gradient 'WithoutGradient, 'Gradient 'WithGradient] :: [Gradient
--                                                      RequiresGradient]
-- = '[ 'Gradient 'WithoutGradient, 'Gradient 'WithGradient]
-- >>> :kind! GetGradients ('Just ('Gradient 'WithGradient))
-- GetGradients ('Just ('Gradient 'WithGradient)) :: [Gradient RequiresGradient]
-- = '[ 'Gradient 'WithGradient]
type GetGradients :: k -> [Gradient RequiresGradient]
type family GetGradients f where
  GetGradients (a :: Gradient RequiresGradient) = '[a]
  GetGradients (f g) = Concat (GetGradients f) (GetGradients g)
  GetGradients _ = '[]
