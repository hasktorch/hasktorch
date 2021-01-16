{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.RequiresGradient where

import Type.Errors.Pretty (ToErrorMessage)

-- | Data type to represent whether or not the tensor requires gradient computation.
data RequiresGradient
  = -- | The tensor requires gradients. We say that the tensor is independent and thus a leaf in the computation graph.
    Independent
  | -- | Gradient computations for this tensor are disabled. We say that the tensor is dependent.
    Dependent
  deriving (Show, Eq)

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'Independent where
  requiresGradientVal = Independent

instance KnownRequiresGradient 'Dependent where
  requiresGradientVal = Dependent
