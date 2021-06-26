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
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2 #-}

module Torch.GraduallyTyped.NN.Normalization where

import Control.Monad.State.Strict (MonadState (state), runState)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF, layerNormWithBias, layerNormWithoutBias)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), KnownShape, Name (..), SShape (..), Shape (..), Size (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (withoutCreate), ones, randn, sOnes, sZeros, zeros)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  LayerNorm
    (hasBias :: HasBias)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (normalizedShape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  LayerNormWithBias ::
    forall device dataType normalizedShape.
    { layerNormWithBiasWeight :: Tensor 'WithGradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormBias :: Tensor 'WithGradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormWithBiasEps :: Double
    } ->
    LayerNorm 'WithBias device dataType normalizedShape
  LayerNormWithoutBias ::
    forall device dataType normalizedShape.
    { layerNormWithoutBiasWeight :: Tensor 'WithGradient ('Layout 'Dense) device dataType normalizedShape,
      layerNormWithoutBiasEps :: Double
    } ->
    LayerNorm 'WithoutBias device dataType normalizedShape

instance HasInitialize (LayerNorm 'WithBias device dataType normalizedShape) where
  type
    InitializeF (LayerNorm 'WithBias device dataType normalizedShape) =
      SDevice device ->
      SDataType dataType ->
      SShape normalizedShape ->
      Double ->
      LayerNorm 'WithBias device dataType normalizedShape
  initialize device dataType normalizedShape eps =
    let weight = sOnes SWithGradient (SLayout SDense) device dataType normalizedShape
        bias = sZeros SWithGradient (SLayout SDense) device dataType normalizedShape
     in LayerNormWithBias weight bias eps

instance HasInitialize (LayerNorm 'WithoutBias device dataType normalizedShape) where
  type
    InitializeF (LayerNorm 'WithoutBias device dataType normalizedShape) =
      SDevice device ->
      SDataType dataType ->
      SShape normalizedShape ->
      Double ->
      LayerNorm 'WithoutBias device dataType normalizedShape
  initialize device dataType normalizedShape eps =
    let weight =
          sOnes SWithGradient (SLayout SDense) device dataType normalizedShape
     in LayerNormWithoutBias weight eps

instance
  ( KnownShape normalizedShape,
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LayerNormWithBiasF normalizedShape normalizedShape shape')
  ) =>
  HasForward
    (LayerNorm 'WithBias device dataType normalizedShape)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward LayerNormWithBias {..} input g = (layerNormWithBias layerNormWithBiasWeight layerNormBias layerNormWithBiasEps input, g)

instance
  ( KnownShape normalizedShape,
    KnownShape shape',
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> layout')
          (device <+> device')
          (dataType <+> dataType')
          (LayerNormWithoutBiasF normalizedShape shape')
  ) =>
  HasForward
    (LayerNorm 'WithoutBias device dataType normalizedShape)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
    output
    generator
  where
  forward LayerNormWithoutBias {..} input g = (layerNormWithoutBias layerNormWithoutBiasWeight layerNormWithoutBiasEps input, g)
