{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Dropout where

import Data.Kind (Type)
import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
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
  deriving stock (Show, Generic)

instance (Scalar p) => HasInitialize (Dropout p) p generator generator where
  initialize p g = (Dropout p, g)

instance HasStateDict (Dropout p) p where
  fromStateDict p _ = pure $ Dropout p
  toStateDict _ _ = pure ()

instance
  ( Scalar p,
    input ~ Tensor gradient layout device dataType shape,
    generator ~ Generator generatorDevice,
    output ~ Tensor gradient layout (device <+> generatorDevice) dataType shape,
    generatorOutput ~ Generator (device <+> generatorDevice)
  ) =>
  HasForward
    (Dropout p)
    input
    generator
    output
    generatorOutput
  where
  forward (Dropout _p) input g = pure $ (unsafeCoerce :: (input, g) -> (output, generatorOutput)) (input, g)
