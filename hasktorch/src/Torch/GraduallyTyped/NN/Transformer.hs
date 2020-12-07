{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Torch.GraduallyTyped.NN.Transformer where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (type (<=), type (*), KnownNat, Div, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (UnifyDataTypeF, DataType (DataType), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (UnifyDeviceF, Device (..), DeviceType(..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward, forward, HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (Dropout))
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearC, Linear (Linear))
import Torch.GraduallyTyped.Random (generator, Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (BroadcastShapesF, type (!), UnifyDimF, GetDimF, dimSize, Size(..), Name(..), KnownDim(..), WithShapeC(..), NumelF, By(..), SelectDim(..), Dim (..), Shape (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar, add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, reshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Layout
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient(Dependent))
import Torch.GraduallyTyped.Tensor.Creation (randn)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormC, LayerNorm)
import Torch.GraduallyTyped.NN.Functional.Activation (relu)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormF)

--------------------------------------------------------------------------------
-- Multi-Headed Attention Layer
--------------------------------------------------------------------------------

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

type HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
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
        dropout <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        pure $ MultiheadAttention qInProj kInProj vInProj outProj dropout

type MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputGeneratorDevice outputDataType outputShape =
  ( KnownDim embedDim,
    KnownDim headEmbedDim,
    KnownDim headDim,
    KnownDim keySeqDim,
    KnownDim querySeqDim,
    KnownDim batchDim,
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
          ( UnifyLayoutF
              (UnifyLayoutF keyLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, keyEmbedDim])
              ( 'Shape '[embedDim])
              keyShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF keyLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
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
          ( UnifyLayoutF
              (UnifyLayoutF valueLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, valueEmbedDim])
              ( 'Shape '[embedDim])
              valueShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF valueLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
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
          ( UnifyLayoutF
              (UnifyLayoutF queryLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, queryEmbedDim])
              ( 'Shape '[embedDim])
              queryShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF queryLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
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
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( UnifyDeviceF
              ( UnifyDeviceF
                  ( UnifyDeviceF
                      (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
                      (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
                  )
                  generatorDevice
              )
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          ( UnifyDataTypeF
              ( UnifyDataTypeF
                  (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
                  (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
              )
              (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
          )
          ( TransposeF
              ( 'SelectDim ( 'ByIndex 1))
              ( 'SelectDim ( 'ByIndex 2))
              ( MatmulF
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
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( UnifyDeviceF
              ( UnifyDeviceF
                  ( UnifyDeviceF
                      (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
                      (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
                  )
                  generatorDevice
              )
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          ( UnifyDataTypeF
              ( UnifyDataTypeF
                  ( UnifyDataTypeF
                      (UnifyDataTypeF queryDataType dataType)
                      dataType
                  )
                  ( UnifyDataTypeF
                      (UnifyDataTypeF keyDataType dataType)
                      dataType
                  )
              )
              ( UnifyDataTypeF
                  (UnifyDataTypeF valueDataType dataType)
                  dataType
              )
          )
          ( ReshapeF
              ( TransposeF
                  ( 'SelectDim ( 'ByIndex 1))
                  ( 'SelectDim ( 'ByIndex 2))
                  ( MatmulF
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
                                          '[ batchDim,
                                             querySeqDim,
                                             headDim,
                                             headEmbedDim
                                           ]
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
                      ( TransposeF
                          ( 'SelectDim ( 'ByIndex 1))
                          ( 'SelectDim ( 'ByIndex 2))
                          ( ReshapeF
                              ( LinearF
                                  ( 'Shape '[embedDim, valueEmbedDim])
                                  ( 'Shape '[embedDim])
                                  valueShape
                              )
                              ( 'Shape
                                  '[batchDim, keySeqDim, headDim, headEmbedDim]
                              )
                          )
                      )
                  )
              )
              ( 'Shape '[batchDim, querySeqDim, embedDim])
          )
      ),
    batchDim ~ UnifyDimF (UnifyDimF (queryShape ! 0) (keyShape ! 0)) (valueShape ! 0),
    querySeqDim ~ (queryShape ! 1),
    keySeqDim ~ UnifyDimF (keyShape ! 1) (valueShape ! 1)
  )

type MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        (UnifyDeviceF queryDevice device)
                        device
                    )
                    ( UnifyDeviceF
                        (UnifyDeviceF keyDevice device)
                        device
                    )
                )
                generatorDevice
            )
            ( UnifyDeviceF
                (UnifyDeviceF valueDevice device)
                device
            )
        )
        device
    )
    device

type MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
        (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
    )
    generatorDevice

type MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape =
  LinearF
    ( 'Shape '[queryEmbedDim, embedDim])
    ( 'Shape '[queryEmbedDim])
    ( ReshapeF
        ( TransposeF
            ( 'SelectDim ( 'ByIndex 1))
            ( 'SelectDim ( 'ByIndex 2))
            ( MatmulF
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
                                ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
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
                                        '[batchDim, keySeqDim, headDim, headEmbedDim]
                                    )
                                )
                            )
                        )
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
        )
        ( 'Shape '[batchDim, querySeqDim, embedDim])
    )

type MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            ( UnifyDataTypeF
                (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
                (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
            )
            (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
        )
        dataType
    )
    dataType

type MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout = 
  UnifyLayoutF
      ( UnifyLayoutF
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( 'Layout 'Dense)
      )
      ( 'Layout 'Dense)

multiheadAttention ::
  forall headDim headEmbedDim generatorDevice batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape outputLayout outputDevice outputGeneratorDevice outputDataType outputShape.
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputGeneratorDevice outputDataType outputShape,
    outputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout,
    outputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice,
    outputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    outputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType,
    outputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape
  ) =>
  -- | multi-head attention model
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
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
multiheadAttention MultiheadAttention {..} query key value =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let batchDim = case dimVal @batchDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> undefined
          querySeqDim = case dimVal @querySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> undefined
          keySeqDim = case dimVal @keySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> undefined
          embedDim = case dimVal @embedDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> undefined
          scaling :: Double = sqrt . fromIntegral . dimSize $ headDim
          q =
            transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              . reshape' @batchDim @querySeqDim @headDim @headEmbedDim [batchDim, querySeqDim, headDim, headEmbedDim]
              . flip divScalar scaling
              . forward @_ @_ @() mhaQInProj
              $ query
          k =
            transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
              . forward @_ @_ @() mhaKInProj
              $ key
          qk = q `matmul` transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3)) k
          (weights, g') = forward @_ @_ @(Generator generatorDevice) mhaDropout (softmax @( 'SelectDim ( 'ByIndex 3)) qk) g
          v =
            transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
              . forward @_ @_ @() mhaVInProj
              $ value
      in ( forward @_ @_ @() mhaOutProj
              . reshape'' @batchDim @querySeqDim @embedDim [batchDim, querySeqDim, embedDim]
              . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              $ weights `matmul` v,
            g'
          )
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
type TestEmbedDim = 'Dim ('Name "*") ('Size 768)
-- type TestFFNDim = 'Dim ('Name "ffn") ('Size 256)
type TestFFNDim = 'Dim ('Name "*") ('Size 256)
-- type TestQueryEmbedDim = 'Dim ('Name "queryEmbed") ('Size 512)
type TestQueryEmbedDim = 'Dim ('Name "*") ('Size 512)
-- type TestKeyEmbedDim = 'Dim ('Name "keyEmbed") ('Size 2048)
type TestKeyEmbedDim = 'Dim ('Name "*") ('Size 2048)
-- type TestValueEmbedDim = 'Dim ('Name "valueEmbed") ('Size 1024)
type TestValueEmbedDim = 'Dim ('Name "*") ('Size 1024)
-- type TestQuerySeqDim = 'Dim ('Name "querySeq") ('Size 32)
type TestQuerySeqDim = 'Dim ('Name "*") ('Size 32)
-- type TestKeySeqDim = 'Dim ('Name "keySeq") ('Size 48)
type TestKeySeqDim = 'Dim ('Name "*") ('Size 48)
-- type TestBatchDim = 'Dim ('Name "batch") ('Size 4)
type TestBatchDim = 'Dim ('Name "*") ('Size 4)
type TestHeadDim = 'Dim ('Name "head") ('Size 12)
-- type TestHeadDim = 'Dim ('Name "*") ('Size 12)
type TestHeadEmbedDim = 'Dim ('Name "headEmbed") ('Size 64)
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
  g <- generator @TestDevice 0
  let (result, _) = runState
        ( do
            mha <- state $ initialize @(MultiheadAttention TestDevice TestDataType TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0
            query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
            key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
            value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
            state $ multiheadAttention @TestHeadDim @TestHeadEmbedDim mha query key value -- 12 64
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
      tmlpLayerNorm :: LayerNorm device dataType ('Shape '[embedDim])
    } ->
    TransformerMLP device dataType embedDim ffnDim dropoutP

type HasInitializeTransformerMLPC device dataType embedDim ffnDim dropoutP =
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
        dropout0 <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        dropout1 <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        layerNorm <-
          pure $
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

type TransformerMLPOutputLayout inputLayout =
  UnifyLayoutF
    ( UnifyLayoutF
        ( UnifyLayoutF
            inputLayout
            ( UnifyLayoutF
                ( UnifyLayoutF
                    ( UnifyLayoutF
                        (UnifyLayoutF inputLayout ( 'Layout 'Dense))
                        ( 'Layout 'Dense)
                    )
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
        )
        ( 'Layout 'Dense)
    )
    ( 'Layout 'Dense)

type TransformerMLPOutputDevice device inputDevice outputGeneratorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        (UnifyDeviceF inputDevice outputGeneratorDevice)
        device
    )
    device

type TransformerMLPOutputDataType dataType inputDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            inputDataType
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    (UnifyDataTypeF (UnifyDataTypeF inputDataType dataType) dataType)
                    dataType
                )
                dataType
            )
        )
        dataType
    )
    dataType

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

type TransformerMLPOutputGeneratorDevice device inputDevice generatorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                (UnifyDeviceF (UnifyDeviceF inputDevice device) device)
                generatorDevice
            )
            device
        )
        device
    )
    ( UnifyDeviceF
        (UnifyDeviceF (UnifyDeviceF inputDevice device) device)
        generatorDevice
    )

transformerMLP ::
  forall device dataType embedDim ffnDim dropoutP requiresGradient inputLayout inputDevice inputDataType inputShape generatorDevice outputLayout outputDevice outputDataType outputShape outputGeneratorDevice.
  ( Scalar dropoutP,
    KnownDim embedDim,
    outputLayout ~ TransformerMLPOutputLayout inputLayout,
    outputDevice ~ TransformerMLPOutputDevice device inputDevice outputGeneratorDevice,
    outputDataType ~ TransformerMLPOutputDataType dataType inputDataType,
    outputShape ~ TransformerMLPOutputShape embedDim ffnDim inputShape,
    outputGeneratorDevice ~ TransformerMLPOutputGeneratorDevice device inputDevice generatorDevice
  ) =>
  TransformerMLP device dataType embedDim ffnDim dropoutP ->
  Tensor requiresGradient inputLayout inputDevice inputDataType inputShape ->
  Generator generatorDevice ->
  ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
    Generator outputGeneratorDevice
  )
transformerMLP TransformerMLP {..} =
  let residual f f' x g =
        let (x', g') = f x g
         in (f' (x `add` x'), g')
      f x g =
        let (x', g') =
              forward @_ @_ @(Generator generatorDevice)
                tmlpDropout0
                (relu . forward @_ @_ @() tmlpLinear0 $ x)
                g
         in forward @_ @_ @(Generator (UnifyDeviceF (UnifyDeviceF (UnifyDeviceF inputDevice device) device) generatorDevice))
              tmlpDropout1
              (forward @_ @_ @() tmlpLinear1 x')
              g'
   in residual f (forward @_ @_ @() tmlpLayerNorm)
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
  g <- generator @TestDevice 0
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
      tlLayerNorm :: LayerNorm device dataType ('Shape '[queryEmbedDim]),
      -- | MLP
      tlTransformerMLP :: TransformerMLP device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
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
        attentionDropout <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        layerNorm <-
          pure $
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

type TransformerLayerOutputLayout queryLayout multiheadAttentionOutputLayout =
  UnifyLayoutF
    ( UnifyLayoutF
        ( UnifyLayoutF
            ( UnifyLayoutF
                ( UnifyLayoutF
                    (UnifyLayoutF queryLayout multiheadAttentionOutputLayout)
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
            ( UnifyLayoutF
                ( UnifyLayoutF
                    ( UnifyLayoutF
                        ( UnifyLayoutF
                            ( UnifyLayoutF
                                ( UnifyLayoutF
                                    ( UnifyLayoutF
                                        queryLayout
                                        multiheadAttentionOutputLayout
                                    )
                                    ( 'Layout 'Dense)
                                )
                                ( 'Layout 'Dense)
                            )
                            ( 'Layout 'Dense)
                        )
                        ( 'Layout 'Dense)
                    )
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
        )
        ( 'Layout 'Dense)
    )
    ( 'Layout 'Dense)

type TransformerLayerOutputDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice outputGeneratorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        queryDevice
                        ( UnifyDeviceF
                            multiheadAttentionOutputDevice
                            multiheadAttentionOutputGeneratorDevice
                        )
                    )
                    device
                )
                device
            )
            outputGeneratorDevice
        )
        device
    )
    device

type TransformerLayerOutputGeneratorDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice =
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        ( UnifyDeviceF
                            ( UnifyDeviceF
                                ( UnifyDeviceF
                                    queryDevice
                                    ( UnifyDeviceF
                                        multiheadAttentionOutputDevice
                                        multiheadAttentionOutputGeneratorDevice
                                    )
                                )
                                device
                            )
                            device
                        )
                        device
                    )
                    device
                )
                ( UnifyDeviceF
                    multiheadAttentionOutputDevice
                    multiheadAttentionOutputGeneratorDevice
                )
            )
            device
        )
        device
    )
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        ( UnifyDeviceF
                            queryDevice
                            ( UnifyDeviceF
                                multiheadAttentionOutputDevice
                                multiheadAttentionOutputGeneratorDevice
                            )
                        )
                        device
                    )
                    device
                )
                device
            )
            device
        )
        ( UnifyDeviceF
            multiheadAttentionOutputDevice
            multiheadAttentionOutputGeneratorDevice
        )
    )

type TransformerLayerOutputDataType dataType queryDataType multiheadAttentionOutputDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    (UnifyDataTypeF queryDataType multiheadAttentionOutputDataType)
                    dataType
                )
                dataType
            )
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    ( UnifyDataTypeF
                        ( UnifyDataTypeF
                            ( UnifyDataTypeF
                                ( UnifyDataTypeF
                                    ( UnifyDataTypeF
                                        queryDataType
                                        multiheadAttentionOutputDataType
                                    )
                                    dataType
                                )
                                dataType
                            )
                            dataType
                        )
                        dataType
                    )
                    dataType
                )
                dataType
            )
        )
        dataType
    )
    dataType

type TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape =
  LayerNormF
    ( 'Shape '[queryEmbedDim])
    ( 'Shape '[queryEmbedDim])
    ( BroadcastShapesF
        ( LayerNormF
            ( 'Shape '[queryEmbedDim])
            ( 'Shape '[queryEmbedDim])
            (BroadcastShapesF queryShape multiheadAttentionOutputShape)
        )
        ( LinearF
            ( 'Shape '[queryEmbedDim, ffnDim])
            ( 'Shape '[queryEmbedDim])
            ( LinearF
                ( 'Shape '[ffnDim, queryEmbedDim])
                ( 'Shape '[ffnDim])
                ( LayerNormF
                    ( 'Shape '[queryEmbedDim])
                    ( 'Shape '[queryEmbedDim])
                    (BroadcastShapesF queryShape multiheadAttentionOutputShape)
                )
            )
        )
    )

transformerLayer ::
  forall headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputDataType outputShape outputGeneratorDevice multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape.
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape,
    WithDimC headDim (WithDimF headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient outputLayout outputDevice outputDataType outputShape, Generator outputGeneratorDevice))),
    WithDimC headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient outputLayout outputDevice outputDataType outputShape, Generator outputGeneratorDevice)),
    multiheadAttentionOutputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout,
    multiheadAttentionOutputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice,
    multiheadAttentionOutputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    multiheadAttentionOutputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType,
    multiheadAttentionOutputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape,
    KnownDim queryEmbedDim,
    outputShape ~ TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape,
    outputDataType ~ TransformerLayerOutputDataType dataType queryDataType multiheadAttentionOutputDataType,
    outputLayout ~ TransformerLayerOutputLayout queryLayout multiheadAttentionOutputLayout,
    outputDevice ~ TransformerLayerOutputDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice outputGeneratorDevice,
    outputGeneratorDevice ~ TransformerLayerOutputGeneratorDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice
  ) =>
  -- | transformer layer model
  TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
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
transformerLayer TransformerLayer {..} query key value =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let residual f f' query g =
            let (query', g') = f query g
             in (f' (query `add` query'), g')
          f (query :: Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (g :: Generator generatorDevice) =
            let (query' :: Tensor requiresGradient multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape, g' :: (Generator multiheadAttentionOutputGeneratorDevice)) =
                  withoutDim @headEmbedDim
                    ( withoutDim @headDim
                        ( multiheadAttention @headDim @headEmbedDim @generatorDevice tlMultiheadAttention query key value
                        )
                        headDim
                    )
                    headEmbedDim
                    g
             in forward @_ @_ @(Generator multiheadAttentionOutputGeneratorDevice) tlAttentionDropout query' g'
          (query', g') = residual f (forward @_ @_ @() tlLayerNorm) query g
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
  g <- generator @TestDevice 0
  let (result, _) =
        runState
          ( do
              tl <- state $ initialize @(TransformerLayer TestDevice TestDataType TestEmbedDim TestFFNDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0 1e-5
              query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
              key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
              value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
              state $ transformerLayer @TestHeadDim @TestHeadEmbedDim tl query key value
          )
          g
  pure result
