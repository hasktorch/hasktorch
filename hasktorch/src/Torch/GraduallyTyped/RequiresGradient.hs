{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.RequiresGradient where

import Type.Errors.Pretty (ToErrorMessage, TypeError)

-- | Data type to represent whether or not the tensor requires gradient computation.
data RequiresGradient
  = -- | The tensor requires gradients. We say the tensor is independent and thus a leaf in the computation graph.
    Independent
  | -- | The tensor does not require gradients. We say the tensor is dependent.
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