{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL1
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C #-}

module Torch.GraduallyTyped.NN.Transformer where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (DataType), KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Activation (relu)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormF)
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearC, Linear (..))
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormC, LayerNorm)
import Torch.GraduallyTyped.Random (Generator, mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (Dependent))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (BroadcastShapesF, By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..), dimSize, getDim, unifyDims, type (!))
import Torch.GraduallyTyped.Tensor.Creation (randn)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, UnsqueezeF, reshape, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, shape)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  MultiheadAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  MultiheadAttention ::
    { -- | in-projection for query
      mhaQInProj :: Linear device dataType queryEmbedDim embedDim,
      -- | in-projection for key
      mhaKInProj :: Linear device dataType keyEmbedDim embedDim,
      -- | in-projection for value
      mhaVInProj :: Linear device dataType valueEmbedDim embedDim,
      -- | out-projection
      mhaOutProj :: Linear device dataType embedDim queryEmbedDim,
      -- | dropout
      mhaDropout :: Dropout dropoutP
    } ->
    MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeMultiheadAttentionC ::
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  Constraint

type HasInitializeMultiheadAttentionC
  device
  dataType
  embedDim
  queryEmbedDim
  keyEmbedDim
  valueEmbedDim
  dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    WithDimC queryEmbedDim (Generator device -> (Linear device dataType embedDim queryEmbedDim, Generator device)),
    HasInitializeLinearC device dataType queryEmbedDim embedDim,
    HasInitializeLinearC device dataType keyEmbedDim embedDim,
    HasInitializeLinearC device dataType valueEmbedDim embedDim,
    HasInitializeLinearC device dataType embedDim queryEmbedDim,
    Scalar dropoutP
  )

instance
  HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    queryEmbedDim
                    ( WithDimF
                        keyEmbedDim
                        ( WithDimF
                            valueEmbedDim
                            (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
                        )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @embedDim $
              \embedDim ->
                withDim @queryEmbedDim $
                  \queryEmbedDim ->
                    withDim @keyEmbedDim $
                      \keyEmbedDim ->
                        withDim @valueEmbedDim @(dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                          \valueEmbedDim ->
                            go deviceType dType embedDim queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP = runState $ do
        qInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType queryEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              embedDim
        kInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType keyEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  keyEmbedDim
              )
              embedDim
        vInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @valueEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType valueEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  valueEmbedDim
              )
              embedDim
        outProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @embedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType embedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  embedDim
              )
              queryEmbedDim
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ MultiheadAttention qInProj kInProj vInProj outProj dropout

type BatchDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type BatchDim queryShape keyShape valueShape =
  (queryShape ! 0) <+> (keyShape ! 0) <+> (valueShape ! 0)

unsafeGetBatchDim :: [Dim String Integer] -> [Dim String Integer] -> [Dim String Integer] -> Dim String Integer
unsafeGetBatchDim queryDims keyDims valueDims =
  unsafePerformIO $ do
    dim <- getDim (ByIndex 0) queryDims
    dims <- sequence [getDim (ByIndex 0) keyDims, getDim (ByIndex 0) valueDims]
    unifyDims dim dims

type QuerySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type QuerySeqDim queryShape =
  queryShape ! 1

unsafeGetQuerySeqDim :: [Dim String Integer] -> Dim String Integer
unsafeGetQuerySeqDim = unsafePerformIO . getDim (ByIndex 1)

type KeySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type KeySeqDim keyShape valueShape =
  (keyShape ! 1) <+> (valueShape ! 1)

unsafeGetKeySeqDim :: [Dim String Integer] -> [Dim String Integer] -> Dim String Integer
unsafeGetKeySeqDim keyDims valueDims =
  unsafePerformIO $ do
    dim <- getDim (ByIndex 1) keyDims
    dims <- sequence [getDim (ByIndex 1) valueDims]
    unifyDims dim dims

unsafeGetEmbedDim ::
  forall device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
  (KnownDim embedDim, KnownDim queryEmbedDim, KnownDim keyEmbedDim, KnownDim valueEmbedDim) =>
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  Dim String Integer
unsafeGetEmbedDim MultiheadAttention {..} =
  unsafePerformIO $ do
    dim <- getDim (ByIndex 0) . shape . linearWeight $ mhaQInProj
    dims <-
      sequence
        [ getDim (ByIndex 0) . shape . linearBias $ mhaQInProj,
          getDim (ByIndex 0) . shape . linearWeight $ mhaKInProj,
          getDim (ByIndex 0) . shape . linearBias $ mhaKInProj,
          getDim (ByIndex 0) . shape . linearWeight $ mhaVInProj,
          getDim (ByIndex 0) . shape . linearBias $ mhaVInProj,
          getDim (ByIndex 1) . shape . linearWeight $ mhaOutProj
        ]
    unifyDims dim dims

type TransposeAndReshape ::
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]

type TransposeAndReshape
  embedDim
  queryEmbedDim
  queryShape
  batchDim
  querySeqDim
  headDim
  headEmbedDim
  keyEmbedDim
  keyShape
  keySeqDim
  valueEmbedDim
  valueShape
  attentionMaskShape =
  TransposeF
    ( 'SelectDim ( 'ByIndex 1))
    ( 'SelectDim ( 'ByIndex 2))
    ( MatmulF
        ( BroadcastShapesF
            ( SoftmaxF
                ( 'SelectDim ( 'ByIndex 3))
                ( MatmulF
                    ( TransposeF
                        ( 'SelectDim ( 'ByIndex 1))
                        ( 'SelectDim ( 'ByIndex 2))
                        ( ReshapeF
                            ( LinearF
                                ( 'Shape '[embedDim, queryEmbedDim])
                                ( 'Shape '[embedDim])
                                queryShape
                            )
                            ( 'Shape
                                '[batchDim, querySeqDim, headDim, headEmbedDim]
                            )
                        )
                    )
                    ( TransposeF
                        ( 'SelectDim ( 'ByIndex 2))
                        ( 'SelectDim ( 'ByIndex 3))
                        ( TransposeF
                            ( 'SelectDim ( 'ByIndex 1))
                            ( 'SelectDim ( 'ByIndex 2))
                            ( ReshapeF
                                ( LinearF
                                    ( 'Shape '[embedDim, keyEmbedDim])
                                    ( 'Shape '[embedDim])
                                    keyShape
                                )
                                ( 'Shape
                                    '[ batchDim,
                                       keySeqDim,
                                       headDim,
                                       headEmbedDim
                                     ]
                                )
                            )
                        )
                    )
                )
            )
            ( UnsqueezeF
                ( 'SelectDim ( 'ByIndex 1))
                attentionMaskShape
            )
        )
        ( TransposeF
            ( 'SelectDim ( 'ByIndex 1))
            ( 'SelectDim ( 'ByIndex 2))
            ( ReshapeF
                ( LinearF
                    ( 'Shape '[embedDim, valueEmbedDim])
                    ( 'Shape '[embedDim])
                    valueShape
                )
                ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
            )
        )
    )

type MultiheadAttentionC ::
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  RequiresGradient ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Device (DeviceType Nat) ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Constraint

type MultiheadAttentionC
  headDim
  headEmbedDim
  batchDim
  querySeqDim
  keySeqDim
  device
  dataType
  embedDim
  queryEmbedDim
  keyEmbedDim
  valueEmbedDim
  dropoutP
  requiresGradient
  queryLayout
  queryDevice
  queryDataType
  queryShape
  keyLayout
  keyDevice
  keyDataType
  keyShape
  valueLayout
  valueDevice
  valueDataType
  valueShape
  attentionMaskLayout
  attentionMaskDevice
  attentionMaskDataType
  attentionMaskShape
  generatorDevice
  outputLayout
  outputDevice
  outputGeneratorDevice
  outputDataType
  outputShape =
  ( KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim keyEmbedDim,
    KnownDim valueEmbedDim,
    KnownDim headEmbedDim,
    KnownDim headDim,
    KnownDim keySeqDim,
    KnownDim querySeqDim,
    KnownDim batchDim,
    KnownShape queryShape,
    KnownShape keyShape,
    KnownShape valueShape,
    KnownLayout attentionMaskLayout,
    KnownDevice attentionMaskDevice,
    KnownDataType attentionMaskDataType,
    KnownShape attentionMaskShape,
    Scalar dropoutP,
    WithDimC
      headDim
      ( WithDimF
          headEmbedDim
          ( Generator generatorDevice ->
            ( Tensor
                requiresGradient
                outputLayout
                outputDevice
                outputDataType
                outputShape,
              Generator outputGeneratorDevice
            )
          )
      ),
    WithDimC
      headEmbedDim
      ( Generator generatorDevice ->
        ( Tensor
            requiresGradient
            outputLayout
            outputDevice
            outputDataType
            outputShape,
          Generator outputGeneratorDevice
        )
      ),
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( 'Layout 'Dense <+> keyLayout)
          (device <+> keyDevice)
          (dataType <+> keyDataType)
          ( LinearF
              ( 'Shape '[embedDim, keyEmbedDim])
              ( 'Shape '[embedDim])
              keyShape
          ) ->
        Tensor
          requiresGradient
          ( 'Layout 'Dense <+> keyLayout)
          (device <+> keyDevice)
          (dataType <+> keyDataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, keyEmbedDim])
                  ( 'Shape '[embedDim])
                  keyShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( 'Layout 'Dense <+> valueLayout)
          (device <+> valueDevice)
          (dataType <+> valueDataType)
          ( LinearF
              ( 'Shape '[embedDim, valueEmbedDim])
              ( 'Shape '[embedDim])
              valueShape
          ) ->
        Tensor
          requiresGradient
          ( 'Layout 'Dense <+> valueLayout)
          (device <+> valueDevice)
          (dataType <+> valueDataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, valueEmbedDim])
                  ( 'Shape '[embedDim])
                  valueShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( LinearF
              ( 'Shape '[embedDim, queryEmbedDim])
              ( 'Shape '[embedDim])
              queryShape
          ) ->
        Tensor
          requiresGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, queryEmbedDim])
                  ( 'Shape '[embedDim])
                  queryShape
              )
              ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, embedDim])
      ( Tensor
          requiresGradient
          ( 'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType)
          (TransposeAndReshape embedDim queryEmbedDim queryShape batchDim querySeqDim headDim headEmbedDim keyEmbedDim keyShape keySeqDim valueEmbedDim valueShape attentionMaskShape) ->
        Tensor
          requiresGradient
          ( 'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType)
          ( ReshapeF
              (TransposeAndReshape embedDim queryEmbedDim queryShape batchDim querySeqDim headDim headEmbedDim keyEmbedDim keyShape keySeqDim valueEmbedDim valueShape attentionMaskShape)
              ( 'Shape '[batchDim, querySeqDim, embedDim])
          )
      ),
    batchDim ~ BatchDim queryShape keyShape valueShape,
    querySeqDim ~ QuerySeqDim queryShape,
    keySeqDim ~ KeySeqDim keyShape valueShape
  )

type MultiheadAttentionOutputLayout ::
  Layout LayoutType ->
  Layout LayoutType ->
  Layout LayoutType ->
  Layout LayoutType ->
  Layout LayoutType

type MultiheadAttentionOutputLayout
  queryLayout
  keyLayout
  valueLayout
  attentionMaskLayout =
  'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout

type MultiheadAttentionOutputDevice ::
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat)

type MultiheadAttentionOutputDevice
  device
  queryDevice
  keyDevice
  valueDevice
  attentionMaskDevice
  generatorDevice =
  device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice

type MultiheadAttentionOutputDataType ::
  DataType DType ->
  DataType DType ->
  DataType DType ->
  DataType DType ->
  DataType DType ->
  DataType DType

type MultiheadAttentionOutputDataType
  dataType
  queryDataType
  keyDataType
  valueDataType
  attentionMaskDataType =
  dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType

type MultiheadAttentionOutputShape ::
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]

type MultiheadAttentionOutputShape
  embedDim
  queryEmbedDim
  keyEmbedDim
  valueEmbedDim
  headDim
  headEmbedDim
  batchDim
  querySeqDim
  keySeqDim
  queryShape
  keyShape
  valueShape
  attentionMaskShape =
  LinearF
    ( 'Shape '[queryEmbedDim, embedDim])
    ( 'Shape '[queryEmbedDim])
    ( ReshapeF
        (TransposeAndReshape embedDim queryEmbedDim queryShape batchDim querySeqDim headDim headEmbedDim keyEmbedDim keyShape keySeqDim valueEmbedDim valueShape attentionMaskShape)
        ( 'Shape '[batchDim, querySeqDim, embedDim])
    )

type MultiheadAttentionOutputGeneratorDevice ::
  Device
    (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat)

type MultiheadAttentionOutputGeneratorDevice
  device
  queryDevice
  keyDevice
  generatorDevice =
  device <+> queryDevice <+> keyDevice <+> generatorDevice

multiheadAttention ::
  forall
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (generatorDevice :: Device (DeviceType Nat))
    (batchDim :: Dim (Name Symbol) (Size Nat))
    (querySeqDim :: Dim (Name Symbol) (Size Nat))
    (keySeqDim :: Dim (Name Symbol) (Size Nat))
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
    (requiresGradient :: RequiresGradient)
    (queryLayout :: Layout LayoutType)
    (queryDevice :: Device (DeviceType Nat))
    (queryDataType :: DataType DType)
    (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (keyLayout :: Layout LayoutType)
    (keyDevice :: Device (DeviceType Nat))
    (keyDataType :: DataType DType)
    (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (valueLayout :: Layout LayoutType)
    (valueDevice :: Device (DeviceType Nat))
    (valueDataType :: DataType DType)
    (valueShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (attentionMaskLayout :: Layout LayoutType)
    (attentionMaskDevice :: Device (DeviceType Nat))
    (attentionMaskDataType :: DataType DType)
    (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (outputLayout :: Layout LayoutType)
    (outputDevice :: Device (DeviceType Nat))
    (outputGeneratorDevice :: Device (DeviceType Nat))
    (outputDataType :: DataType DType)
    (outputShape :: Shape [Dim (Name Symbol) (Size Nat)]).
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice outputLayout outputDevice outputGeneratorDevice outputDataType outputShape,
    outputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout attentionMaskLayout,
    outputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice,
    outputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    outputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType attentionMaskDataType,
    outputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape attentionMaskShape
  ) =>
  -- | multi-head attention model
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  -- | attention mask
  Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape ->
  WithDimF
    headDim
    ( WithDimF
        headEmbedDim
        ( Generator generatorDevice ->
          ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
            Generator outputGeneratorDevice
          )
        )
    )
multiheadAttention mha@MultiheadAttention {..} query key value attentionMask =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let batchDim = case dimVal @batchDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> unsafeGetBatchDim (shape query) (shape key) (shape value)
          querySeqDim = case dimVal @querySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> unsafeGetQuerySeqDim (shape query)
          keySeqDim = case dimVal @keySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> unsafeGetKeySeqDim (shape key) (shape value)
          embedDim = case dimVal @embedDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> unsafeGetEmbedDim mha
          scaling :: Double = sqrt . fromIntegral . dimSize $ headDim
          (q, g') =
            let (query', g') = forward mhaQInProj query g
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @querySeqDim @headDim @headEmbedDim [batchDim, querySeqDim, headDim, headEmbedDim]
                    . flip divScalar scaling
                    $ query',
                  g'
                )
          (k, g'') =
            let (key', g'') = forward mhaKInProj key g'
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
                    $ key',
                  g''
                )
          qk = q `matmul` transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3)) k
          (weights, g''') = forward @_ @_ @(Generator generatorDevice) mhaDropout (softmax @( 'SelectDim ( 'ByIndex 3)) qk) g''
          weights' = weights `add` unsqueeze @( 'SelectDim ( 'ByIndex 1)) attentionMask
          (v, g'''') =
            let (value', g'''') = forward mhaVInProj value g'''
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
                    $ value',
                  g''''
                )
          weights'' =
            reshape'' @batchDim @querySeqDim @embedDim [batchDim, querySeqDim, embedDim]
              . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              $ weights' `matmul` v
       in forward mhaOutProj weights'' g''''
  where
    reshape' ::
      forall batchDim seqDim headDim headEmbedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
    reshape' [batchDim, seqDim, headDim, headEmbedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
         )
        (reshape @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, headDim, headEmbedDim]
        input
    reshape'' ::
      forall batchDim seqDim embedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, embedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
    reshape'' [batchDim, seqDim, embedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, embedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
         )
        (reshape @( 'Shape '[batchDim, seqDim, embedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, embedDim]
        input

type TestDevice :: Device (DeviceType Nat)

type TestDevice = 'Device 'CPU

type TestLayout = 'Layout 'Dense

type TestDataType = 'DataType 'Float

-- type TestEmbedDim = 'Dim ('Name "embed") ('Size 768)
type TestEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

-- type TestFFNDim = 'Dim ('Name "ffn") ('Size 256)
type TestFFNDim = 'Dim ( 'Name "*") ( 'Size 256)

-- type TestQueryEmbedDim = 'Dim ('Name "queryEmbed") ('Size 512)
type TestQueryEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- type TestKeyEmbedDim = 'Dim ('Name "keyEmbed") ('Size 2048)
type TestKeyEmbedDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- type TestValueEmbedDim = 'Dim ('Name "valueEmbed") ('Size 1024)
type TestValueEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

-- type TestQuerySeqDim = 'Dim ('Name "querySeq") ('Size 32)
type TestQuerySeqDim = 'Dim ( 'Name "*") ( 'Size 32)

-- type TestKeySeqDim = 'Dim ('Name "keySeq") ('Size 48)
type TestKeySeqDim = 'Dim ( 'Name "*") ( 'Size 48)

-- type TestBatchDim = 'Dim ('Name "batch") ('Size 4)
type TestBatchDim = 'Dim ( 'Name "*") ( 'Size 4)

type TestHeadDim = 'Dim ( 'Name "head") ( 'Size 12)

-- type TestHeadDim = 'Dim ('Name "*") ('Size 12)
type TestHeadEmbedDim = 'Dim ( 'Name "headEmbed") ( 'Size 64)

-- type TestHeadEmbedDim = 'Dim ('Name "*") ('Size 64)
-- type TestHeadDim :: Dim (Name Symbol) (Size Nat)
-- type TestHeadDim = 'Dim ('Name "head") 'UncheckedSize
-- type TestHeadEmbedDim :: Dim (Name Symbol) (Size Nat)
-- type TestHeadEmbedDim = 'Dim ('Name "headEmbed") 'UncheckedSize

testmha ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
        -- 'UncheckedShape
    )
testmha = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              mha <- state $ initialize @(MultiheadAttention TestDevice TestDataType TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0
              query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
              key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
              value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
              attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
              state $ multiheadAttention @TestHeadDim @TestHeadEmbedDim mha query key value attentionMask -- 12 64
          )
          g
  pure result

data
  TransformerMLP
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerMLP ::
    forall device dataType embedDim ffnDim dropoutP.
    { -- | first fully connected layer
      tmlpLinear0 :: Linear device dataType embedDim ffnDim,
      -- | second fully connected layer
      tmlpLinear1 :: Linear device dataType ffnDim embedDim,
      -- | relu dropout
      tmlpDropout0 :: Dropout dropoutP,
      -- | other dropout
      tmlpDropout1 :: Dropout dropoutP,
      -- | layer norm
      tmlpLayerNorm :: LayerNorm device dataType ( 'Shape '[embedDim])
    } ->
    TransformerMLP device dataType embedDim ffnDim dropoutP

type HasInitializeTransformerMLPC ::
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  Constraint

type HasInitializeTransformerMLPC
  device
  dataType
  embedDim
  ffnDim
  dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)))),
    WithDimC embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)),
    HasInitializeLinearC device dataType embedDim ffnDim,
    HasInitializeLinearC device dataType ffnDim embedDim,
    HasInitializeLayerNormC device dataType ( 'Shape '[embedDim]),
    Scalar dropoutP
  )

instance
  HasInitializeTransformerMLPC device dataType embedDim ffnDim dropoutP =>
  HasInitialize (TransformerMLP device dataType embedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerMLP device dataType embedDim ffnDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    ffnDim
                    ( dropoutP ->
                      Double ->
                      Generator device ->
                      ( TransformerMLP device dataType embedDim ffnDim dropoutP,
                        Generator device
                      )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @embedDim $
              \embedDim ->
                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)) $
                  \ffnDim ->
                    go deviceType dType embedDim ffnDim
    where
      go deviceType dType embedDim ffnDim dropoutP eps = runState $ do
        linear0 <-
          state $
            withoutDim @ffnDim
              ( withoutDim @embedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType embedDim ffnDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  embedDim
              )
              ffnDim
        linear1 <-
          state $
            withoutDim @embedDim
              ( withoutDim @ffnDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType ffnDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  ffnDim
              )
              embedDim
        let dropout0 = initialize @(Dropout dropoutP) dropoutP
        let dropout1 = initialize @(Dropout dropoutP) dropoutP
        let layerNorm =
              withoutShape @( 'Shape '[embedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm device dataType ( 'Shape '[embedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [embedDim]
                eps
        pure $ TransformerMLP linear0 linear1 dropout0 dropout1 layerNorm

type TransformerMLPOutputLayout :: Layout LayoutType -> Layout LayoutType

type TransformerMLPOutputLayout inputLayout =
  'Layout 'Dense <+> inputLayout

type TransformerMLPOutputDevice ::
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat)

type TransformerMLPOutputDevice device inputDevice generatorDevice =
  device <+> inputDevice <+> generatorDevice

type TransformerMLPOutputDataType ::
  DataType DType ->
  DataType DType ->
  DataType DType

type TransformerMLPOutputDataType dataType inputDataType =
  dataType <+> inputDataType

type TransformerMLPOutputShape ::
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)]

type TransformerMLPOutputShape embedDim ffnDim inputShape =
  LayerNormF
    ( 'Shape '[embedDim])
    ( 'Shape '[embedDim])
    ( BroadcastShapesF
        inputShape
        ( LinearF
            ( 'Shape '[embedDim, ffnDim])
            ( 'Shape '[embedDim])
            ( LinearF
                ( 'Shape '[ffnDim, embedDim])
                ( 'Shape '[ffnDim])
                inputShape
            )
        )
    )

type TransformerMLPOutputGeneratorDevice ::
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat) ->
  Device (DeviceType Nat)

type TransformerMLPOutputGeneratorDevice
  device
  inputDevice
  generatorDevice =
  device <+> inputDevice <+> generatorDevice

transformerMLP ::
  forall
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
    (requiresGradient :: RequiresGradient)
    (inputLayout :: Layout LayoutType)
    (inputDevice :: Device (DeviceType Nat))
    (inputDataType :: DataType DType)
    (inputShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (generatorDevice :: Device (DeviceType Nat)).
  ( Scalar dropoutP,
    KnownDim embedDim
  ) =>
  TransformerMLP device dataType embedDim ffnDim dropoutP ->
  Tensor requiresGradient inputLayout inputDevice inputDataType inputShape ->
  Generator generatorDevice ->
  ( Tensor
      requiresGradient
      (TransformerMLPOutputLayout inputLayout)
      (TransformerMLPOutputDevice device inputDevice generatorDevice)
      (TransformerMLPOutputDataType dataType inputDataType)
      (TransformerMLPOutputShape embedDim ffnDim inputShape),
    Generator (TransformerMLPOutputGeneratorDevice device inputDevice generatorDevice)
  )
transformerMLP TransformerMLP {..} =
  let residual f f' x g =
        let (x', g') = f x g
         in f' (x `add` x') g'
      f x g =
        let (x', g') = forward tmlpLinear0 x g
            (x'', g'') = forward tmlpDropout0 (relu x') g'
            (x''', g''') = forward tmlpLinear1 x'' g''
         in forward tmlpDropout1 x''' g'''
   in residual f (forward tmlpLayerNorm)

testmlp ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestEmbedDim])
    )
testmlp = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              mlp <- state $ initialize @(TransformerMLP TestDevice TestDataType TestEmbedDim TestFFNDim Float) 0.0 1e-5
              x <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestEmbedDim])
              state $ transformerMLP mlp x
          )
          g
  pure result

data
  TransformerLayer
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerLayer ::
    forall device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
    { -- | multi-head attention
      tlMultiheadAttention :: MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP,
      -- | dropout
      tlAttentionDropout :: Dropout dropoutP,
      -- | layer norm
      tlLayerNorm :: LayerNorm device dataType ( 'Shape '[queryEmbedDim]),
      -- | MLP
      tlTransformerMLP :: TransformerMLP device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeTransformerLayerC ::
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  Constraint

type HasInitializeTransformerLayerC
  device
  dataType
  embedDim
  ffnDim
  queryEmbedDim
  keyEmbedDim
  valueEmbedDim
  dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDimC ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP,
    HasInitializeLayerNormC device dataType ( 'Shape '[queryEmbedDim]),
    HasInitializeTransformerMLPC device dataType queryEmbedDim ffnDim dropoutP,
    Scalar dropoutP
  )

instance
  HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    ffnDim
                    ( WithDimF
                        queryEmbedDim
                        ( WithDimF
                            keyEmbedDim
                            ( WithDimF
                                valueEmbedDim
                                (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
                            )
                        )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @embedDim $
              \embedDim ->
                withDim @ffnDim $
                  \ffnDim ->
                    withDim @queryEmbedDim $
                      \queryEmbedDim ->
                        withDim @keyEmbedDim $
                          \keyEmbedDim ->
                            withDim @valueEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                              \valueEmbedDim ->
                                go deviceType dType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP eps = runState $ do
        multiheadAttention <-
          state $
            withoutDim @valueEmbedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDataType @dataType
                              ( withoutDevice @device
                                  ( initialize @(MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
                                  )
                                  deviceType
                              )
                              dType
                          )
                          embedDim
                      )
                      queryEmbedDim
                  )
                  keyEmbedDim
              )
              valueEmbedDim
              dropoutP
        let attentionDropout = initialize @(Dropout dropoutP) dropoutP
        let layerNorm =
              withoutShape @( 'Shape '[queryEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm device dataType ( 'Shape '[queryEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
        transformerMLP <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(TransformerMLP device dataType queryEmbedDim ffnDim dropoutP)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              ffnDim
              dropoutP
              eps
        pure $ TransformerLayer multiheadAttention attentionDropout layerNorm transformerMLP

type TransformerLayerOutputLayout (multiheadAttentionOutputLayout :: Layout LayoutType) = multiheadAttentionOutputLayout

type TransformerLayerOutputLayout' queryLayout keyLayout valueLayout attentionMaskLayout = TransformerLayerOutputLayout (MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout attentionMaskLayout)

type TransformerLayerOutputDevice (multiheadAttentionOutputDevice :: Device (DeviceType Nat)) = multiheadAttentionOutputDevice

type TransformerLayerOutputDevice' device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice = TransformerLayerOutputDevice (MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice)

type TransformerLayerOutputGeneratorDevice (multiheadAttentionOutputDevice :: Device (DeviceType Nat)) = multiheadAttentionOutputDevice

type TransformerLayerOutputGeneratorDevice' device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice = TransformerLayerOutputGeneratorDevice (MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice)

type TransformerLayerOutputDataType (multiheadAttentionOutputDataType :: DataType DType) = multiheadAttentionOutputDataType

type TransformerLayerOutputDataType' dataType queryDataType keyDataType valueDataType attentionMaskDataType = TransformerLayerOutputDataType (MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType attentionMaskDataType)

type TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape =
  TransformerMLPOutputShape
    queryEmbedDim
    ffnDim
    ( LayerNormF
        ( 'Shape '[queryEmbedDim])
        ( 'Shape '[queryEmbedDim])
        (BroadcastShapesF queryShape multiheadAttentionOutputShape)
    )

type TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim queryShape keyShape valueShape attentionMaskShape = TransformerLayerOutputShape ffnDim queryEmbedDim queryShape (MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim (BatchDim queryShape keyShape valueShape) (QuerySeqDim queryShape) (KeySeqDim keyShape valueShape) queryShape keyShape valueShape attentionMaskShape)

transformerLayer ::
  forall
    headDim
    headEmbedDim
    batchDim
    querySeqDim
    keySeqDim
    device
    dataType
    embedDim
    ffnDim
    queryEmbedDim
    keyEmbedDim
    valueEmbedDim
    dropoutP
    requiresGradient
    queryLayout
    queryDevice
    queryDataType
    queryShape
    keyLayout
    keyDevice
    keyDataType
    keyShape
    valueLayout
    valueDevice
    valueDataType
    valueShape
    attentionMaskLayout
    attentionMaskDevice
    attentionMaskDataType
    attentionMaskShape
    generatorDevice
    outputLayout
    outputDevice
    outputDataType
    outputShape
    outputGeneratorDevice
    multiheadAttentionOutputLayout
    multiheadAttentionOutputDevice
    multiheadAttentionOutputGeneratorDevice
    multiheadAttentionOutputDataType
    multiheadAttentionOutputShape.
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape,
    multiheadAttentionOutputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout attentionMaskLayout,
    multiheadAttentionOutputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice attentionMaskDevice generatorDevice,
    multiheadAttentionOutputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    multiheadAttentionOutputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType attentionMaskDataType,
    multiheadAttentionOutputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape attentionMaskShape,
    KnownDim queryEmbedDim,
    WithDimC
      headEmbedDim
      ( Generator generatorDevice ->
        ( Tensor
            requiresGradient
            (TransformerLayerOutputLayout multiheadAttentionOutputLayout)
            (TransformerLayerOutputDevice multiheadAttentionOutputDevice)
            (TransformerLayerOutputDataType multiheadAttentionOutputDataType)
            (TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape),
          Generator (TransformerLayerOutputGeneratorDevice multiheadAttentionOutputDevice)
        )
      ),
    WithDimC
      headDim
      ( WithDimF
          headEmbedDim
          ( Generator generatorDevice ->
            ( Tensor
                requiresGradient
                (TransformerLayerOutputLayout multiheadAttentionOutputLayout)
                (TransformerLayerOutputDevice multiheadAttentionOutputDevice)
                (TransformerLayerOutputDataType multiheadAttentionOutputDataType)
                (TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape),
              Generator (TransformerLayerOutputGeneratorDevice multiheadAttentionOutputDevice)
            )
          )
      )
      -- outputLayout ~ multiheadAttentionOutputLayout,
      -- outputDevice ~ multiheadAttentionOutputDevice,
      -- outputDataType ~ multiheadAttentionOutputDataType,
      -- outputShape ~ TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape
      -- outputGeneratorDevice ~ multiheadAttentionOutputDevice
  ) =>
  -- | transformer layer model
  TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  -- | attention mask
  Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape ->
  WithDimF
    headDim
    ( WithDimF
        headEmbedDim
        ( Generator generatorDevice ->
          ( Tensor
              requiresGradient
              (TransformerLayerOutputLayout multiheadAttentionOutputLayout)
              (TransformerLayerOutputDevice multiheadAttentionOutputDevice)
              (TransformerLayerOutputDataType multiheadAttentionOutputDataType)
              (TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape),
            Generator (TransformerLayerOutputGeneratorDevice multiheadAttentionOutputDevice)
          )
        )
    )
transformerLayer TransformerLayer {..} query key value attentionMask =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let residual f f' query g =
            let (query', g') = f query g
             in f' (query `add` query') g'
          f (query :: Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (g :: Generator generatorDevice) =
            let (query' :: Tensor requiresGradient multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape, g' :: (Generator multiheadAttentionOutputGeneratorDevice)) =
                  withoutDim @headEmbedDim
                    ( withoutDim @headDim
                        ( multiheadAttention @headDim @headEmbedDim @generatorDevice tlMultiheadAttention query key value attentionMask
                        )
                        headDim
                    )
                    headEmbedDim
                    g
             in forward tlAttentionDropout query' g'
          (query', g') = residual f (forward tlLayerNorm) query g
       in transformerMLP tlTransformerMLP query' g'

testtl ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
    )
testtl = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              tl <- state $ initialize @(TransformerLayer TestDevice TestDataType TestEmbedDim TestFFNDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0 1e-5
              query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
              key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
              value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
              attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestKeySeqDim])
              state $ transformerLayer @TestHeadDim @TestHeadEmbedDim tl query key value attentionMask
          )
          g
  pure result

data
  TransformerLayerStack
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerLayerStackNil ::
    TransformerLayerStack 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  TransformerLayerStackCons ::
    -- | head dim
    Dim String Integer ->
    -- | head embed dim
    Dim String Integer ->
    TransformerLayer device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP ->
    TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP ->
    TransformerLayerStack (numLayers + 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP

type HasInitializeTransformerLayerStack ::
  Bool ->
  Nat ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  Constraint
class
  HasInitializeTransformerLayerStack
    nil
    numLayers
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    ffnDim
    queryEmbedDim
    dropoutP
  where
  initializeTransformerLayerStack ::
    Proxy nil ->
    WithDeviceF
      device
      ( WithDataTypeF
          dataType
          ( WithDimF
              headDim
              ( WithDimF
                  headEmbedDim
                  ( WithDimF
                      embedDim
                      ( WithDimF
                          ffnDim
                          ( WithDimF
                              queryEmbedDim
                              (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerLayerStackC ::
  Nat ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  Constraint

type HasInitializeTransformerLayerStackC
  numLayers
  device
  dataType
  headDim
  headEmbedDim
  embedDim
  ffnDim
  queryEmbedDim
  dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))),
    WithDimC queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerLayerStackC 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP =>
  HasInitializeTransformerLayerStack 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  where
  initializeTransformerLayerStack _ =
    withDevice @device $
      \_deviceType ->
        withDataType @dataType $
          \_dType ->
            withDim @headDim $
              \_headDim ->
                withDim @headEmbedDim $
                  \_headEmbedDim ->
                    withDim @embedDim $
                      \_embedDim ->
                        withDim @ffnDim $
                          \_ffnDim ->
                            withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)) $
                              \_queryEmbedDim _dropoutP _eps g -> (TransformerLayerStackNil, g)

instance
  ( HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    HasInitializeTransformerLayerStackC (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP,
    HasInitializeTransformerLayerStackC numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP,
    HasInitialize (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
  ) =>
  HasInitializeTransformerLayerStack 'True numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  where
  initializeTransformerLayerStack _ =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @ffnDim $
                          \ffnDim ->
                            withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)) $
                              \queryEmbedDim ->
                                go deviceType dType headDim headEmbedDim embedDim ffnDim queryEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP eps = runState $ do
        layerStack <-
          state $
            withoutDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
              ( withoutDim @ffnDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
                                      )
                                      deviceType
                                  )
                                  dType
                              )
                              headDim
                          )
                          headEmbedDim
                      )
                      embedDim
                  )
                  ffnDim
              )
              queryEmbedDim
              dropoutP
              eps
        layer <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @ffnDim
                          ( withoutDim @embedDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerLayer device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
                                      )
                                      deviceType
                                  )
                                  dType
                              )
                              embedDim
                          )
                          ffnDim
                      )
                      queryEmbedDim
                  )
                  queryEmbedDim
              )
              queryEmbedDim
              dropoutP
              eps
        pure $ TransformerLayerStackCons headDim headEmbedDim layer layerStack

instance
  HasInitializeTransformerLayerStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP =>
  HasInitialize (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( WithDimF
                        embedDim
                        ( WithDimF
                            ffnDim
                            ( WithDimF
                                queryEmbedDim
                                (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerLayerStack @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @ffnDim @queryEmbedDim @dropoutP (Proxy :: Proxy (1 <=? numLayers))

type HasForwardTransformerLayerStack ::
  Bool ->
  Nat ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type ->
  RequiresGradient ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Layout LayoutType ->
  Device (DeviceType Nat) ->
  DataType DType ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Device (DeviceType Nat) ->
  Constraint
class
  HasForwardTransformerLayerStack
    nil
    numLayers
    headDim
    headEmbedDim
    device
    dataType
    embedDim
    ffnDim
    queryEmbedDim
    dropoutP
    requiresGradient
    queryLayout
    queryDevice
    queryDataType
    queryShape
    attentionMaskLayout
    attentionMaskDevice
    attentionMaskDataType
    attentionMaskShape
    generatorDevice
  where
  type HasForwardTransformerLayerStackOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice :: Type
  type HasForwardTransformerLayerStackGeneratorOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice :: Type
  forwardTransformerLayerStack ::
    Proxy nil ->
    TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP ->
    ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    ) ->
    Generator generatorDevice ->
    ( HasForwardTransformerLayerStackOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice,
      HasForwardTransformerLayerStackGeneratorOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
    )

instance
  ( WithDimC headDim (WithDimF headEmbedDim (HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice)),
    WithDimC headEmbedDim (HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice)
  ) =>
  HasForwardTransformerLayerStack 'False 0 headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  where
  type HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice = Tensor requiresGradient queryLayout queryDevice queryDataType queryShape
  type HasForwardTransformerLayerStackGeneratorOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice = Generator generatorDevice
  forwardTransformerLayerStack _ TransformerLayerStackNil (x, _attentionMask) g = (x, g)

instance
  ( MultiheadAttentionC
      headDim
      headEmbedDim
      (BatchDim queryShape queryShape queryShape)
      (QuerySeqDim queryShape)
      (QuerySeqDim queryShape)
      device
      dataType
      embedDim
      queryEmbedDim
      queryEmbedDim
      queryEmbedDim
      dropoutP
      requiresGradient
      queryLayout
      queryDevice
      queryDataType
      queryShape
      queryLayout
      queryDevice
      queryDataType
      queryShape
      queryLayout
      queryDevice
      queryDataType
      queryShape
      attentionMaskLayout
      attentionMaskDevice
      attentionMaskDataType
      attentionMaskShape
      generatorDevice
      -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
      (MultiheadAttentionOutputLayout queryLayout queryLayout queryLayout attentionMaskLayout)
      -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
      (MultiheadAttentionOutputDevice device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
      -- (UnifyDeviceF queryDevice (UnifyDeviceF device generatorDevice))
      (MultiheadAttentionOutputGeneratorDevice device queryDevice queryDevice generatorDevice)
      -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
      (MultiheadAttentionOutputDataType dataType queryDataType queryDataType queryDataType attentionMaskDataType)
      (MultiheadAttentionOutputShape embedDim queryEmbedDim queryEmbedDim queryEmbedDim headDim headEmbedDim (BatchDim queryShape queryShape queryShape) (QuerySeqDim queryShape) (QuerySeqDim queryShape) queryShape queryShape queryShape attentionMaskShape),
    WithDimC
      headDim
      ( WithDimF
          headEmbedDim
          ( Generator generatorDevice ->
            ( Tensor
                requiresGradient
                -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
                (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
                -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
                (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
                -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
                (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
                (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape),
              Generator
                -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
                (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
            )
          )
      ),
    WithDimC
      headEmbedDim
      ( Generator generatorDevice ->
        ( Tensor
            requiresGradient
            -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
            (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
            -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
            (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
            -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
            (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
            (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape),
          Generator
            -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
            (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
        )
      ),
    -- KnownDim queryEmbedDim,
    HasForward
      (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
      ( Tensor
          requiresGradient
          -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
          (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
          -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
          (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
          -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
          (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
          (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape),
        Tensor
          requiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          attentionMaskShape
      )
      ( Generator
          -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
          (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
      )
  ) =>
  HasForwardTransformerLayerStack 'True numLayers headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  where
  type
    HasForwardTransformerLayerStackOutput
      'True
      numLayers
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      ffnDim
      queryEmbedDim
      dropoutP
      requiresGradient
      queryLayout
      queryDevice
      queryDataType
      queryShape
      attentionMaskLayout
      attentionMaskDevice
      attentionMaskDataType
      attentionMaskShape
      generatorDevice =
      HasForwardTransformerLayerStackOutput
        (1 <=? numLayers - 1)
        (numLayers - 1)
        device
        dataType
        headDim
        headEmbedDim
        embedDim
        ffnDim
        queryEmbedDim
        dropoutP
        requiresGradient
        -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
        (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
        -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
        (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
        -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
        (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
        (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape)
        attentionMaskLayout
        attentionMaskDevice
        attentionMaskDataType
        attentionMaskShape
        (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
  type
    HasForwardTransformerLayerStackGeneratorOutput
      'True
      numLayers
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      ffnDim
      queryEmbedDim
      dropoutP
      requiresGradient
      queryLayout
      queryDevice
      queryDataType
      queryShape
      attentionMaskLayout
      attentionMaskDevice
      attentionMaskDataType
      attentionMaskShape
      generatorDevice =
      HasForwardTransformerLayerStackGeneratorOutput
        (1 <=? numLayers - 1)
        (numLayers - 1)
        device
        dataType
        headDim
        headEmbedDim
        embedDim
        ffnDim
        queryEmbedDim
        dropoutP
        requiresGradient
        -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
        (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
        -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
        (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
        -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
        (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
        (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape)
        attentionMaskLayout
        attentionMaskDevice
        attentionMaskDataType
        attentionMaskShape
        (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
  forwardTransformerLayerStack _ (TransformerLayerStackCons headDim headEmbedDim layer layerStack) (x, attentionMask) g =
    let (x', g') =
          withoutDim @headEmbedDim
            ( withoutDim @headDim
                ( transformerLayer @headDim @headEmbedDim @(BatchDim queryShape queryShape queryShape) @(QuerySeqDim queryShape) @(QuerySeqDim queryShape) @device @dataType @embedDim @ffnDim @queryEmbedDim @queryEmbedDim @queryEmbedDim @dropoutP @requiresGradient @queryLayout @queryDevice @queryDataType @queryShape @queryLayout @queryDevice @queryDataType @queryShape @queryLayout @queryDevice @queryDataType @queryShape @attentionMaskLayout @attentionMaskDevice @attentionMaskDataType @attentionMaskShape @generatorDevice layer x x x attentionMask
                )
                headDim
            )
            headEmbedDim
            g
     in forward
          @(TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
          @( Tensor
               requiresGradient
               -- (UnifyLayoutF queryLayout (UnifyLayoutF ( 'Layout 'Dense) attentionMaskLayout))
               (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout attentionMaskLayout)
               -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
               (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
               -- (UnifyDataTypeF queryDataType (UnifyDataTypeF dataType attentionMaskDataType))
               (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType attentionMaskDataType)
               (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape attentionMaskShape),
             Tensor
               requiresGradient
               attentionMaskLayout
               attentionMaskDevice
               attentionMaskDataType
               attentionMaskShape
           )
          @( Generator
               -- (UnifyDeviceF queryDevice (UnifyDeviceF device (UnifyDeviceF generatorDevice attentionMaskDevice)))
               (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice attentionMaskDevice generatorDevice)
           )
          layerStack
          (x', attentionMask)
          g'

instance
  ( HasForwardTransformerLayerStack (1 <=? numLayers) numLayers headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  ) =>
  HasForward (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice)
  where
  type
    ForwardOutput (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice) =
      HasForwardTransformerLayerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  type
    ForwardGeneratorOutput (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice) =
      HasForwardTransformerLayerStackGeneratorOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  forward = forwardTransformerLayerStack (Proxy @(1 <=? numLayers))

testtlstack ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape _)
    )
testtlstack = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              tlstack <- state $ initialize @(TransformerLayerStack 2 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestFFNDim TestQueryEmbedDim Float) 0.0 1e-5
              x <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, 'Dim 'UncheckedName 'UncheckedSize]) (Dim "*" 512)
              attentionMask <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQuerySeqDim])
              state $ forward tlstack (x, attentionMask)
          )
          g
  pure result