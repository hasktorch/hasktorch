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
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Normalization where

import Control.Monad.State.Strict (MonadState (state), runState)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), UnifyDataTypeF, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), UnifyDeviceF, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense), UnifyLayoutF)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormF, layerNorm)
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), KnownShape, Name (..), Shape (..), Size (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (withoutCreate), ones, randn, zeros)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data
  LayerNorm
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (normalizedShape :: Shape [Dim (Name Symbol) (Size Nat)])
  where
  LayerNorm ::
    forall device dataType normalizedShape.
    { layerNormWeight :: Tensor 'Independent ( 'Layout 'Dense) device dataType normalizedShape,
      layerNormBias :: Tensor 'Independent ( 'Layout 'Dense) device dataType normalizedShape,
      layerNormEps :: Double
    } ->
    LayerNorm device dataType normalizedShape

type HasInitializeLayerNormC device dataType normalizedShape =
  ( WithDeviceC device (WithDataTypeF dataType (WithShapeF normalizedShape (Double -> LayerNorm device dataType normalizedShape))),
    WithDataTypeC dataType (WithShapeF normalizedShape (Double -> LayerNorm device dataType normalizedShape)),
    WithShapeC normalizedShape (Double -> LayerNorm device dataType normalizedShape),
    WithCreateC (Tensor 'Independent ( 'Layout 'Dense) device dataType normalizedShape) 'Independent ( 'Layout 'Dense) device dataType normalizedShape
  )

instance
  HasInitializeLayerNormC device dataType normalizedShape =>
  HasInitialize (LayerNorm device dataType normalizedShape)
  where
  type
    InitializeF (LayerNorm device dataType normalizedShape) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithShapeF
                normalizedShape
                (Double -> LayerNorm device dataType normalizedShape)
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withShape @normalizedShape @(Double -> LayerNorm device dataType normalizedShape) $
              \dims ->
                go deviceType dType dims
    where
      go deviceType dType dims eps =
        let weight =
              withoutCreate @_ @ 'Independent @( 'Layout 'Dense) @device @dataType @normalizedShape
                (ones @ 'Independent @( 'Layout 'Dense) @device @dataType @normalizedShape)
                Independent
                Dense
                deviceType
                dType
                dims
            bias =
              withoutCreate @_ @ 'Independent @( 'Layout 'Dense) @device @dataType @normalizedShape
                (zeros @ 'Independent @( 'Layout 'Dense) @device @dataType @normalizedShape)
                Independent
                Dense
                deviceType
                dType
                dims
         in LayerNorm weight bias eps

instance
  KnownShape normalizedShape =>
  HasForward
    (LayerNorm device dataType normalizedShape)
    (Tensor requiresGradient' layout' device' dataType' shape')
    generator
  where
  type
    ForwardOutput
      (LayerNorm device dataType normalizedShape)
      (Tensor requiresGradient' layout' device' dataType' shape')
      generator =
      ( Tensor
          requiresGradient'
          (UnifyLayoutF (UnifyLayoutF layout' ( 'Layout 'Dense)) ( 'Layout 'Dense))
          (UnifyDeviceF (UnifyDeviceF device' device) device)
          (UnifyDataTypeF (UnifyDataTypeF dataType' dataType) dataType)
          (LayerNormF normalizedShape normalizedShape shape')
      )
  forward LayerNorm {..} = layerNorm layerNormWeight layerNormBias layerNormEps

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
        'Dependent
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
              ln <- pure $ initialize @(LayerNorm TestLayerNormDevice TestLayerNormDataType ( 'Shape '[TestLayerNormFstFeatureDim, TestLayerNormSndFeatureDim])) 1e-5
              input <- state $ randn @ 'Dependent @TestLayerNormLayout @TestLayerNormDevice @TestLayerNormDataType @( 'Shape '[TestLayerNormBatchDim, TestLayerNormFstFeatureDim, TestLayerNormSndFeatureDim])
              pure $ forward ln input
          )
          g
  pure result
