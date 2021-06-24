{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Activation where

import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Activation (gelu, relu)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Shape (By, SSelectDim, SelectDim, WithSelectDimC (..), WithSelectDimF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (tanh)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Prelude hiding (tanh)

data Softmax (selectDim :: SelectDim (By Symbol Nat)) where
  Softmax ::
    forall selectDim.
    {softmaxSelectDim :: SSelectDim selectDim} ->
    Softmax selectDim
  deriving (Generic)

instance HasInitialize (Softmax selectDim) where
  type InitializeF (Softmax selectDim) = SSelectDim selectDim -> Softmax selectDim
  initialize = Softmax

instance
  (output ~ Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)) =>
  HasForward
    (Softmax selectDim)
    (Tensor requiresGradient layout device dataType shape)
    generator
    output
    generator
  where
  forward Softmax {..} input g = (softmax softmaxSelectDim input, g)

data Relu where Relu :: Relu

instance HasInitialize Relu where
  type InitializeF Relu = Relu
  initialize = Relu

instance
  HasForward
    Relu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Relu input g = (relu input, g)

data Gelu where Gelu :: Gelu

instance HasInitialize Gelu where
  type InitializeF Gelu = Gelu
  initialize = Gelu

instance
  HasForward
    Gelu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Gelu input g = (gelu input, g)

data Tanh where Tanh :: Tanh

instance HasInitialize Tanh where
  type InitializeF Tanh = Tanh
  initialize = Tanh

instance
  HasForward
    Tanh
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Tanh input g = (tanh input, g)