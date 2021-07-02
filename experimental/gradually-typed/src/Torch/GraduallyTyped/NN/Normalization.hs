{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Normalization where

import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF, layerNormWithBias, layerNormWithoutBias)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), SShape (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (sOnes, sZeros)
import Torch.GraduallyTyped.Tensor.Type (SGetShape, Tensor)
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

instance
  HasInitialize
    (LayerNorm 'WithBias gradient device dataType normalizedShape)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SShape normalizedShape,
      Double
    )
    generator
    generator
  where
  initialize (gradient, device, dataType, normalizedShape, eps) =
    let weight = sOnes gradient (SLayout SDense) device dataType normalizedShape
        bias = sZeros gradient (SLayout SDense) device dataType normalizedShape
     in (LayerNormWithBias weight bias eps,)

instance
  HasInitialize
    (LayerNorm 'WithoutBias gradient device dataType normalizedShape)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SShape normalizedShape,
      Double
    )
    generator
    generator
  where
  initialize (gradient, device, dataType, normalizedShape, eps) =
    let weight = sOnes gradient (SLayout SDense) device dataType normalizedShape
     in (LayerNormWithoutBias weight eps,)

instance
  HasStateDict
    (LayerNorm 'WithBias gradient device dataType normalizedShape)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SShape normalizedShape,
      Double
    )
  where
  fromStateDict (gradient, device, dataType, normalizedShape, eps) k =
    LayerNormWithBias
      <$> fromStateDict (gradient, SLayout SDense, device, dataType, normalizedShape) (k <> "weight")
      <*> fromStateDict (gradient, SLayout SDense, device, dataType, normalizedShape) (k <> "bias")
      <*> pure eps
  toStateDict k LayerNormWithBias {..} = do
    toStateDict (k <> "weight") layerNormWithBiasWeight
    toStateDict (k <> "bias") layerNormBias

instance
  HasStateDict
    (LayerNorm 'WithoutBias gradient device dataType normalizedShape)
    ( SGradient gradient,
      SDevice device,
      SDataType dataType,
      SShape normalizedShape,
      Double
    )
  where
  fromStateDict (gradient, device, dataType, normalizedShape, eps) k =
    LayerNormWithoutBias
      <$> fromStateDict (gradient, SLayout SDense, device, dataType, normalizedShape) (k <> "weight")
      <*> pure eps
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
    generator
    output
    generator
  where
  forward LayerNormWithBias {..} input = (layerNormWithBias layerNormWithBiasWeight layerNormBias layerNormWithBiasEps input,)

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
    generator
    output
    generator
  where
  forward LayerNormWithoutBias {..} input = (layerNormWithoutBias layerNormWithoutBiasWeight layerNormWithoutBiasEps input,)
