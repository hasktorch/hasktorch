{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GPooler where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType, DataType, SDataType)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Activation (Tanh)
import Torch.GraduallyTyped.NN.Class (HasForward (..), ModelSpec, NamedModel)
import Torch.GraduallyTyped.NN.Linear (GLinearF)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle, TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim, Name, SDim, Size)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data
  GPooler
    (dense :: Type)
    (activation :: Type)
  where
  GPooler ::
    forall dense activation.
    { poolerDense :: dense,
      poolerActivation :: activation
    } ->
    GPooler dense activation

type instance
  ModelSpec (GPooler dense activation) =
    GPooler (ModelSpec dense) (ModelSpec activation)

poolerSpec ::
  forall style gradient device dataType inputEmbedDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim inputEmbedDim ->
  GPooler
    (PoolerDenseF style gradient device dataType inputEmbedDim)
    (PoolerActivationF style)
poolerSpec style gradient device dataType inputEmbedDim = undefined

type family
  PoolerDenseF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  PoolerDenseF 'RoBERTa gradient device dataType inputEmbedDim =
    NamedModel (GLinearF 'WithBias gradient device dataType inputEmbedDim inputEmbedDim)

type family
  PoolerActivationF
    (style :: TransformerStyle) ::
    Type
  where
  PoolerActivationF 'RoBERTa = Tanh

instance
  ( HasForward
      dense
      (Tensor gradient layout device dataType shape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      activation
      tensor0
      generatorDevice0
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GPooler dense activation)
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GPooler {..} input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . forward poolerDense
        >>>= IxStateT . forward poolerActivation
