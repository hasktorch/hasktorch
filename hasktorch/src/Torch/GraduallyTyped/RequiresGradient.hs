{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.RequiresGradient where

-- | Data type to represent whether or not the tensor requires gradient computation.
data RequiresGradient
  = -- | The tensor requires gradients.
    WithGradient
  | -- | Gradient computations for this tensor are disabled.
    WithoutGradient
  deriving (Show, Eq)

class KnownRequiresGradient (requiresGradient :: RequiresGradient) where
  requiresGradientVal :: RequiresGradient

instance KnownRequiresGradient 'WithGradient where
  requiresGradientVal = WithGradient

instance KnownRequiresGradient 'WithoutGradient where
  requiresGradientVal = WithoutGradient