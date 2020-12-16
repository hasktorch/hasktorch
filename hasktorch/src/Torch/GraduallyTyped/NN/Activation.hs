{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Activation where

import Data.Singletons.TypeLits (Nat, Symbol)
import GHC.Generics (Generic)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Shape (By, SelectDim, WithSelectDimC (..), WithSelectDimF)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data Softmax (selectDim :: SelectDim (By Symbol Nat)) where
  Softmax ::
    forall selectDim.
    By String Integer ->
    Softmax selectDim
  deriving (Generic)

instance WithSelectDimC selectDim (Softmax selectDim) => HasInitialize (Softmax selectDim) where
  type InitializeF (Softmax selectDim) = WithSelectDimF selectDim (Softmax selectDim)
  initialize = withSelectDim @selectDim $ Softmax @selectDim

instance
  ( WithSelectDimC
      selectDim
      ( Tensor requiresGradient layout device dataType shape ->
        Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
      )
  ) =>
  HasForward
    (Softmax selectDim)
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  type
    ForwardOutput
      (Softmax selectDim)
      (Tensor requiresGradient layout device dataType shape)
      generator =
      Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)
  type
    ForwardGeneratorOutput
      (Softmax selectDim)
      (Tensor requiresGradient layout device dataType shape)
      generator =
      generator
  forward (Softmax by) input g = (withoutSelectDim @selectDim (softmax @selectDim @requiresGradient @layout @device @dataType @shape) by input, g)
