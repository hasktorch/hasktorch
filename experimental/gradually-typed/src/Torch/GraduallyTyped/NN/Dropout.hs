{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Dropout where

import Data.Kind (Type)
import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))
import Unsafe.Coerce (unsafeCoerce)

-- | Given a random generator, randomly zeroes some of the elements of
-- the input tensor with probability 'p' using samples from a Bernoulli distribution.
-- Each channel will be zeroed out independently on every 'forward' call.
newtype Dropout (p :: Type) where
  Dropout ::
    forall p.
    -- | probability of an element to be zeroed
    p ->
    Dropout p
  deriving (Generic)

instance (Scalar p) => HasInitialize (Dropout p) where
  type InitializeF (Dropout p) = p -> Dropout p
  initialize p = Dropout p

instance
  ( Scalar p,
    input ~ Tensor requiresGradient layout device dataType shape,
    generator ~ Generator generatorDevice,
    output ~ Tensor requiresGradient layout (device <+> generatorDevice) dataType shape,
    generatorOutput ~ Generator (device <+> generatorDevice)
  ) =>
  HasForward
    (Dropout p)
    input
    generator
    output
    generatorOutput
  where
  forward (Dropout _p) input g = unsafeCoerce (input, g)