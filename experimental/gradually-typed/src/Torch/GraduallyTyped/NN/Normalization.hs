{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Normalization where

import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF, layerNormWithBias, layerNormWithoutBias)
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), SShape (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (sOnes, sZeros)
import Torch.GraduallyTyped.Tensor.Type (SGetShape, Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data
  LayerNorm
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (normalizedShape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  LayerNormWithBias ::
    forall gradient device dataType normalizedShape.
    { layerNormWithBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormBias :: Tensor gradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormWithBiasEps :: Double
    } ->
    LayerNorm 'WithBias gradient device dataType normalizedShape
  LayerNormWithoutBias ::
    forall gradient device dataType normalizedShape.
    { layerNormWithoutBiasWeight :: Tensor gradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormWithoutBiasEps :: Double
    } ->
    LayerNorm 'WithoutBias gradient device dataType normalizedShape

data
  LayerNormSpec
    (hasBias :: HasBias)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (normalizedShape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  LayerNormSpec ::
    forall hasBias gradient device dataType normalizedShape.
    SHasBias hasBias ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SShape normalizedShape ->
    Double ->
    LayerNormSpec hasBias gradient device dataType normalizedShape
  deriving stock (Show, Generic)

type instance ModelSpec (LayerNorm hasBias gradient device dataType normalizedShape) = LayerNormSpec hasBias gradient device dataType normalizedShape

instance
  HasInitialize
    (LayerNorm hasBias gradient device dataType normalizedShape)
    generatorDevice
    (LayerNorm hasBias gradient device dataType normalizedShape)
    generatorDevice
  where
  initialize (LayerNormSpec SWithBias gradient device dataType normalizedShape eps) g = do
    let tensorSpec = TensorSpec gradient (SLayout SDense) device dataType normalizedShape
    weight <- sOnes tensorSpec
    bias <- sZeros tensorSpec
    pure (LayerNormWithBias weight bias eps, g)
  initialize (LayerNormSpec SWithoutBias gradient device dataType normalizedShape eps) g = do
    let tensorSpec = TensorSpec gradient (SLayout SDense) device dataType normalizedShape
    weight <- sOnes tensorSpec
    pure (LayerNormWithoutBias weight eps, g)

instance
  HasStateDict
    (LayerNorm hasBias gradient device dataType normalizedShape)
  where
  fromStateDict (LayerNormSpec SWithBias gradient device dataType normalizedShape eps) k =
    LayerNormWithBias
      <$> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType normalizedShape) (k <> "weight")
      <*> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType normalizedShape) (k <> "bias")
      <*> pure eps
  fromStateDict (LayerNormSpec SWithoutBias gradient device dataType normalizedShape eps) k =
    LayerNormWithoutBias
      <$> fromStateDict (TensorSpec gradient (SLayout SDense) device dataType normalizedShape) (k <> "weight")
      <*> pure eps
  toStateDict k LayerNormWithBias {..} = do
    toStateDict (k <> "weight") layerNormWithBiasWeight
    toStateDict (k <> "bias") layerNormBias
  toStateDict k LayerNormWithoutBias {..} =
    toStateDict (k <> "weight") layerNormWithoutBiasWeight

instance
  ( SGetShape normalizedShape,
    output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LayerNormWithBiasF normalizedShape normalizedShape shape')
  ) =>
  HasForward
    (LayerNorm 'WithBias gradient device dataType normalizedShape)
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward LayerNormWithBias {..} input = pure . (layerNormWithBias layerNormWithBiasWeight layerNormBias layerNormWithBiasEps input,)

instance
  ( SGetShape normalizedShape,
    SGetShape shape',
    output
      ~ Tensor
          (gradient <|> gradient')
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LayerNormWithoutBiasF normalizedShape shape')
  ) =>
  HasForward
    (LayerNorm 'WithoutBias gradient device dataType normalizedShape)
    (Tensor gradient' layout' device' dataType' shape')
    generatorDevice
    output
    generatorDevice
  where
  forward LayerNormWithoutBias {..} input = pure . (layerNormWithoutBias layerNormWithoutBiasWeight layerNormWithoutBiasEps input,)
