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
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF, layerNormWithBias, layerNormWithoutBias)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), KnownShape, Name (..), Shape (..), Size (..), WithSelectDimsC, WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (withoutCreate), ones, randn, zeros)
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
    { layerNormWithBiasWeight :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape,
      layerNormBias :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape,
      layerNormWithBiasEps :: Double
    } ->
    LayerNorm 'WithBias device dataType normalizedShape
  LayerNormWithoutBias ::
    forall device dataType normalizedShape.
    { layerNormWithoutBiasWeight :: Tensor 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape,
      layerNormWithoutBiasEps :: Double
    } ->
    LayerNorm 'WithoutBias device dataType normalizedShape

type HasInitializeLayerNormWithBiasC device dataType normalizedShape =
  ( WithDeviceC device (WithDataTypeF dataType (WithShapeF normalizedShape (Double -> LayerNorm 'WithBias device dataType normalizedShape))),
    WithDataTypeC dataType (WithShapeF normalizedShape (Double -> LayerNorm 'WithBias device dataType normalizedShape)),
    WithShapeC normalizedShape (Double -> LayerNorm 'WithBias device dataType normalizedShape),
    WithCreateC (Tensor 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape) 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape
  )

instance
  HasInitializeLayerNormWithBiasC device dataType normalizedShape =>
  HasInitialize (LayerNorm 'WithBias device dataType normalizedShape)
  where
  type
    InitializeF (LayerNorm 'WithBias device dataType normalizedShape) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithShapeF
                normalizedShape
                (Double -> LayerNorm 'WithBias device dataType normalizedShape)
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withShape @normalizedShape @(Double -> LayerNorm 'WithBias device dataType normalizedShape) $
              \dims ->
                go deviceType dType dims
    where
      go deviceType dType dims eps =
        let weight =
              withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape
                (ones @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape)
                WithGradient
                Dense
                deviceType
                dType
                dims
            bias =
              withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape
                (zeros @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape)
                WithGradient
                Dense
                deviceType
                dType
                dims
         in LayerNormWithBias weight bias eps

type HasInitializeLayerNormWithoutBiasC device dataType normalizedShape =
  ( WithDeviceC device (WithDataTypeF dataType (WithShapeF normalizedShape (Double -> LayerNorm 'WithoutBias device dataType normalizedShape))),
    WithDataTypeC dataType (WithShapeF normalizedShape (Double -> LayerNorm 'WithoutBias device dataType normalizedShape)),
    WithShapeC normalizedShape (Double -> LayerNorm 'WithoutBias device dataType normalizedShape),
    WithCreateC (Tensor 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape) 'WithGradient ( 'Layout 'Dense) device dataType normalizedShape
  )

instance
  HasInitializeLayerNormWithoutBiasC device dataType normalizedShape =>
  HasInitialize (LayerNorm 'WithoutBias device dataType normalizedShape)
  where
  type
    InitializeF (LayerNorm 'WithoutBias device dataType normalizedShape) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithShapeF
                normalizedShape
                (Double -> LayerNorm 'WithoutBias device dataType normalizedShape)
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withShape @normalizedShape @(Double -> LayerNorm 'WithoutBias device dataType normalizedShape) $
              \dims ->
                go deviceType dType dims
    where
      go deviceType dType dims eps =
        let weight =
              withoutCreate @_ @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape
                (ones @ 'WithGradient @( 'Layout 'Dense) @device @dataType @normalizedShape)
                WithGradient
                Dense
                deviceType
                dType
                dims
         in LayerNormWithoutBias weight eps

instance
  ( KnownShape normalizedShape,
    output
      ~ Tensor
          'WithGradient
          ( 'Layout 'Dense <+> layout')
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
          ( 'Layout 'Dense <+> layout')
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

type TestLayerNormDevice :: Device (DeviceType Nat)

type TestLayerNormDevice = 'Device 'CPU

type TestLayerNormLayout = 'Layout 'Dense

type TestLayerNormDataType = 'DataType 'Float

type TestLayerNormBatchDim = 'Dim ( 'Name "batch") ( 'Size 4)

-- type TestLayerNormBatchDim = 'Dim ( 'Name "*") ( 'Size 4)

type TestLayerNormFstFeatureDim = 'Dim ( 'Name "fstFeature") ( 'Size 12)

-- type TestLayerNormFstFeatureDim = 'Dim ( 'Name "*") ( 'Size 12)

type TestLayerNormSndFeatureDim = 'Dim ( 'Name "sndFeature") ( 'Size 16)

-- type TestLayerNormSndFeatureDim = 'Dim ( 'Name "*") ( 'Size 16)

testln ::
  IO
    ( Tensor
        'WithGradient
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape
            '[ TestLayerNormBatchDim,
               TestLayerNormFstFeatureDim,
               TestLayerNormSndFeatureDim
             ]
        )
    )
testln = do
  g <- mkGenerator @TestLayerNormDevice 0
  let (result, _) =
        runState
          ( do
              let ln = initialize @(LayerNorm 'WithBias TestLayerNormDevice TestLayerNormDataType ( 'Shape '[TestLayerNormFstFeatureDim, TestLayerNormSndFeatureDim])) 1e-5
              input <- state $ randn @ 'WithoutGradient @TestLayerNormLayout @TestLayerNormDevice @TestLayerNormDataType @( 'Shape '[TestLayerNormBatchDim, TestLayerNormFstFeatureDim, TestLayerNormSndFeatureDim])
              state $ forward ln input
          )
          g
  pure result
