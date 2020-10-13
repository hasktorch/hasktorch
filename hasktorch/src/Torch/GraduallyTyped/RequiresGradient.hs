{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.RequiresGradient where

import Type.Errors.Pretty (ToErrorMessage, TypeError)

data RequiresGradient
  = -- | The tensor requires gradients.
    Independent
  | -- | The tensor does not require gradients.
    Dependent
  deriving (Show, Eq)

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'Independent where
  requiresGradientVal = Independent

instance KnownRequiresGradient 'Dependent where
  requiresGradientVal = Dependent

type family UnifyRequiresGradientF (requiresGradient :: RequiresGradient) (requiresGradient' :: RequiresGradient) :: RequiresGradient where
  UnifyRequiresGradientF requiresGradient requiresGradient = requiresGradient
  UnifyRequiresGradientF _ _ = TypeError (ToErrorMessage "The supplied tensors must all either require or disable gradient calculation.")