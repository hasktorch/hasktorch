{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Dropout where

import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Dropout (dropout)
import Torch.GraduallyTyped.Random (SGetGeneratorDevice)
import Torch.GraduallyTyped.Tensor.Type (SGetDevice, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Given a random generator, randomly zeroes some of the elements of
-- the input tensor with probability 'p' using samples from a Bernoulli distribution.
-- Each channel will be zeroed out independently on every 'forward' call.
newtype Dropout where
  Dropout ::
    -- | probability of an element to be zeroed
    Double ->
    Dropout
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec Dropout = Dropout

instance
  HasInitialize
    Dropout
    generatorDevice
    Dropout
    generatorDevice
  where
  initialize spec = pure . (spec,)

instance HasStateDict Dropout where
  fromStateDict spec _ = pure spec
  toStateDict _ _ = pure ()

instance
  ( input ~ Tensor gradient layout device dataType shape,
    output ~ Tensor gradient layout (device <+> generatorDevice) dataType shape,
    generatorOutputDevice ~ (device <+> generatorDevice),
    SGetDevice device,
    SGetGeneratorDevice generatorDevice
  ) =>
  HasForward
    Dropout
    input
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (Dropout p) input g = dropout p input g
