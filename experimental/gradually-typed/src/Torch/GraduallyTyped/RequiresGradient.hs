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

module Torch.GraduallyTyped.RequiresGradient where

import Data.Kind (Type)
import Data.Singletons (Sing)
import Data.Singletons.Prelude.Check (Check, SCheck (..), type SChecked, type SUnchecked)
import Data.Singletons.TH (genSingletons)
import Torch.GraduallyTyped.Prelude (Concat)

-- | Data type to represent whether or not the tensor requires gradient computations.
data RequiresGradient
  = -- | The tensor requires gradient computations.
    WithGradient
  | -- | Gradient computations for this tensor are disabled.
    WithoutGradient
  deriving (Show, Eq)

genSingletons [''RequiresGradient]

deriving stock instance Show (SRequiresGradient (requiresGradient :: RequiresGradient))

type SGradient :: RequiresGradient -> Type

type SGradient requiresGradient = SChecked requiresGradient

pattern SGradient :: forall (a :: RequiresGradient). Sing a -> SGradient a
pattern SGradient requiresGradient = SChecked requiresGradient

type SUncheckedGradient :: Type

type SUncheckedGradient = SUnchecked RequiresGradient

pattern SUncheckedGradient :: RequiresGradient -> SUncheckedGradient
pattern SUncheckedGradient requiresGradient = SUnchecked requiresGradient

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
type GetGradients :: k -> [Check RequiresGradient Type]
type family GetGradients f where
  GetGradients (a :: Check RequiresGradient Type) = '[a]
  GetGradients (f g) = Concat (GetGradients f) (GetGradients g)
  GetGradients _ = '[]
