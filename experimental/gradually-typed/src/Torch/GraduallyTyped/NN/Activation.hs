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
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Functional.Activation (gelu, geluNew, relu)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Shape (By, SSelectDim, SelectDim)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (tanh)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Prelude hiding (tanh)

data Softmax (selectDim :: SelectDim (By Symbol Nat)) (device :: Device (DeviceType Nat)) where
  Softmax ::
    forall selectDim device.
    {softmaxSelectDim :: SSelectDim selectDim} ->
    Softmax selectDim device
  deriving (Generic)

instance HasInitialize (Softmax selectDim) (SSelectDim selectDim) generator generator where
  initialize selectDim = (Softmax selectDim,)

instance HasStateDict (Softmax selectDim device) (SSelectDim selectDim) where
  fromStateDict selectDim _ = pure (Softmax selectDim)
  toStateDict _ _ = pure ()

instance
  (output ~ Tensor requiresGradient layout device dataType (SoftmaxF selectDim shape)) =>
  HasForward
    (Softmax selectDim device)
    (Tensor requiresGradient layout device dataType shape)
    generator
    output
    generator
  where
  forward Softmax {..} input = pure . (softmax softmaxSelectDim input,)

data Relu where Relu :: Relu

instance HasInitialize Relu () generator generator where
  initialize _ = (Relu,)

instance HasStateDict Relu () where
  fromStateDict _ _ = pure Relu
  toStateDict _ _ = pure ()

instance
  HasForward
    Relu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Relu input = pure . (relu input,)

data Gelu where Gelu :: Gelu

instance HasInitialize Gelu () generator generator where
  initialize _ = (Gelu,)

instance HasStateDict Gelu () where
  fromStateDict _ _ = pure Gelu
  toStateDict _ _ = pure ()

instance
  HasForward
    Gelu
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Gelu input = pure . (gelu input,)

data GeluNew where GeluNew :: GeluNew

instance HasInitialize GeluNew () generator generator where
  initialize _ = (GeluNew,)

instance HasStateDict GeluNew () where
  fromStateDict _ _ = pure GeluNew
  toStateDict _ _ = pure ()

instance
  HasForward
    GeluNew
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward GeluNew input = pure . (geluNew input,)

data Tanh where Tanh :: Tanh

instance HasInitialize Tanh () generator generator where
  initialize _ = (Tanh,)

instance HasStateDict Tanh () where
  fromStateDict _ _ = pure Tanh
  toStateDict _ _ = pure ()

instance
  HasForward
    Tanh
    (Tensor requiresGradient layout device dataType shape)
    generator
    (Tensor requiresGradient layout device dataType shape)
    generator
  where
  forward Tanh input = pure . (tanh input,)