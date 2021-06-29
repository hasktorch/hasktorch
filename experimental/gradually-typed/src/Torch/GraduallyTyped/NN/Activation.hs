{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Activation where

import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Activation (gelu, geluNew, relu)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Shape (By, SSelectDim, SelectDim)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (tanh)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Prelude hiding (tanh)

data Softmax (selectDim :: SelectDim (By Symbol Nat)) where
  Softmax ::
    forall selectDim.
    {softmaxSelectDim :: SSelectDim selectDim} ->
    Softmax selectDim
  deriving (Generic)

instance HasInitialize (Softmax selectDim) (SSelectDim selectDim) generator generator where
  initialize selectDim = (Softmax selectDim,)

instance
  (output ~ Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)) =>
  HasForward
    (Softmax selectDim)
    (Tensor requiresGradient layout device dataType shape)
    generator
    output
    generator
  where
  forward Softmax {..} input = (softmax softmaxSelectDim input,)

data Relu where Relu :: Relu

instance HasInitialize Relu () generator generator where
  initialize _ = (Relu,)

instance
  HasForward
    Relu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Relu input = (relu input,)

data Gelu where Gelu :: Gelu

instance HasInitialize Gelu () generator generator where
  initialize _ = (Gelu,)

instance
  HasForward
    Gelu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Gelu input = (gelu input,)

data GeluNew where GeluNew :: GeluNew

instance HasInitialize GeluNew () generator generator where
  initialize _ = (GeluNew,)

instance
  HasForward
    GeluNew
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward GeluNew input = (geluNew input,)

data Tanh where Tanh :: Tanh

instance HasInitialize Tanh () generator generator where
  initialize _ = (Tanh,)

instance
  HasForward
    Tanh
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Tanh input = (tanh input,)