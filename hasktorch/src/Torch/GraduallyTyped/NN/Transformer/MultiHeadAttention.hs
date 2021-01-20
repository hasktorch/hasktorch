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
{-# OPTIONS_GHC -fomit-interface-pragmas
                -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearWithoutBiasC, Linear (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (BroadcastShapesF, By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..), dimSize, getDim, unifyDims, type (!))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, UnsqueezeF, reshape, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, shape)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  MultiHeadAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  MultiHeadAttention ::
    forall device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
    { -- | head dim
      mhaHeadDim :: Dim String Integer,
      -- | head embed dim
      mhaHeadEmbedDim :: Dim String Integer,
      -- | in-projection for query
      mhaQInProj :: Linear 'WithoutBias device dataType queryEmbedDim embedDim,
      -- | in-projection for key
      mhaKInProj :: Linear 'WithoutBias device dataType keyEmbedDim embedDim,
      -- | in-projection for value
      mhaVInProj :: Linear 'WithoutBias device dataType valueEmbedDim embedDim,
      -- | out-projection
      mhaOutProj :: Linear 'WithoutBias device dataType embedDim queryEmbedDim,
      -- | dropout
      mhaDropout :: Dropout dropoutP
    } ->
    MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeMultiHeadAttentionC
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    WithDimC queryEmbedDim (Generator device -> (Linear 'WithBias device dataType embedDim queryEmbedDim, Generator device)),
    HasInitializeLinearWithoutBiasC device dataType queryEmbedDim embedDim,
    HasInitializeLinearWithoutBiasC device dataType keyEmbedDim embedDim,
    HasInitializeLinearWithoutBiasC device dataType valueEmbedDim embedDim,
    HasInitializeLinearWithoutBiasC device dataType embedDim queryEmbedDim,
    Scalar dropoutP
  )

instance
  HasInitializeMultiHeadAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
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
                            queryEmbedDim
                            ( WithDimF
                                keyEmbedDim
                                ( WithDimF
                                    valueEmbedDim
                                    (dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
                                )
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
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @queryEmbedDim $
                          \queryEmbedDim ->
                            withDim @keyEmbedDim $
                              \keyEmbedDim ->
                                withDim @valueEmbedDim @(dropoutP -> Generator device -> (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                                  \valueEmbedDim ->
                                    go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP = runState $ do
        qInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear 'WithoutBias device dataType queryEmbedDim embedDim)
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
                          ( initialize @(Linear 'WithoutBias device dataType keyEmbedDim embedDim)
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
                          ( initialize @(Linear 'WithoutBias device dataType valueEmbedDim embedDim)
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
                          ( initialize @(Linear 'WithoutBias device dataType embedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  embedDim
              )
              queryEmbedDim
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ MultiHeadAttention headDim headEmbedDim qInProj kInProj vInProj outProj dropout

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
  forall device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
  (KnownDim embedDim, KnownDim queryEmbedDim, KnownDim keyEmbedDim, KnownDim valueEmbedDim) =>
  MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  Dim String Integer
unsafeGetEmbedDim MultiHeadAttention {..} =
  unsafePerformIO $ do
    dim <- getDim (ByIndex 0) . shape . linearWithoutBiasWeight $ mhaQInProj
    dims <-
      sequence
        [ getDim (ByIndex 0) . shape . linearWithoutBiasWeight $ mhaKInProj,
          getDim (ByIndex 0) . shape . linearWithoutBiasWeight $ mhaVInProj,
          getDim (ByIndex 1) . shape . linearWithoutBiasWeight $ mhaOutProj
        ]
    unifyDims dim dims

type TransposeAndReshape
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
  (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (valueShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (batchDim :: Dim (Name Symbol) (Size Nat))
  (querySeqDim :: Dim (Name Symbol) (Size Nat))
  (keySeqDim :: Dim (Name Symbol) (Size Nat)) =
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
                            ( LinearWithoutBiasF
                                ( 'Shape '[embedDim, queryEmbedDim])
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
                                ( LinearWithoutBiasF
                                    ( 'Shape '[embedDim, keyEmbedDim])
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
                ( LinearWithoutBiasF
                    ( 'Shape '[embedDim, valueEmbedDim])
                    valueShape
                )
                ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
            )
        )
    )

type MultiHeadAttentionOutputShape
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (batchDim :: Dim (Name Symbol) (Size Nat))
  (querySeqDim :: Dim (Name Symbol) (Size Nat))
  (transposedAndReshaped :: Shape [Dim (Name Symbol) (Size Nat)]) =
  LinearWithoutBiasF
    ( 'Shape '[queryEmbedDim, embedDim])
    ( ReshapeF
        transposedAndReshaped
        ( 'Shape '[batchDim, querySeqDim, embedDim])
    )

type HasForwardMultiHeadAttentionC
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type)
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
  (generatorDevice :: Device (DeviceType Nat))
  (batchDim :: Dim (Name Symbol) (Size Nat))
  (querySeqDim :: Dim (Name Symbol) (Size Nat))
  (keySeqDim :: Dim (Name Symbol) (Size Nat))
  (transposedAndReshaped :: Shape [Dim (Name Symbol) (Size Nat)]) =
  ( KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim keyEmbedDim,
    KnownDim valueEmbedDim,
    KnownDim keySeqDim,
    KnownDim querySeqDim,
    KnownDim batchDim,
    KnownShape queryShape,
    KnownShape keyShape,
    KnownShape valueShape,
    Scalar dropoutP,
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> keyLayout)
          (device <+> keyDevice)
          (dataType <+> keyDataType)
          ( LinearWithoutBiasF
              ( 'Shape '[embedDim, keyEmbedDim])
              keyShape
          ) ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> keyLayout)
          (device <+> keyDevice)
          (dataType <+> keyDataType)
          ( ReshapeF
              ( LinearWithoutBiasF
                  ( 'Shape '[embedDim, keyEmbedDim])
                  keyShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> valueLayout)
          (device <+> valueDevice)
          (dataType <+> valueDataType)
          ( LinearWithoutBiasF
              ( 'Shape '[embedDim, valueEmbedDim])
              valueShape
          ) ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> valueLayout)
          (device <+> valueDevice)
          (dataType <+> valueDataType)
          ( ReshapeF
              ( LinearWithoutBiasF
                  ( 'Shape '[embedDim, valueEmbedDim])
                  valueShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( LinearWithoutBiasF
              ( 'Shape '[embedDim, queryEmbedDim])
              queryShape
          ) ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( ReshapeF
              ( LinearWithoutBiasF
                  ( 'Shape '[embedDim, queryEmbedDim])
                  queryShape
              )
              ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, embedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType)
          transposedAndReshaped ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType)
          ( ReshapeF
              transposedAndReshaped
              ( 'Shape '[batchDim, querySeqDim, embedDim])
          )
      )
  )

instance
  ( HasForwardMultiHeadAttentionC
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      queryEmbedDim
      keyEmbedDim
      valueEmbedDim
      dropoutP
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
      batchDim
      querySeqDim
      keySeqDim
      transposedAndReshaped,
    batchDim ~ BatchDim queryShape keyShape valueShape,
    querySeqDim ~ QuerySeqDim queryShape,
    keySeqDim ~ KeySeqDim keyShape valueShape,
    transposedAndReshaped ~ TransposeAndReshape headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim queryShape keyShape valueShape attentionMaskShape batchDim querySeqDim keySeqDim
  ) =>
  HasForward
    (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      Tensor
        'WithGradient
        ( 'Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionMaskLayout <+> valueLayout)
        (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice <+> valueDevice)
        (dataType <+> queryDataType <+> keyDataType <+> attentionMaskDataType <+> valueDataType)
        ( MultiHeadAttentionOutputShape
            embedDim
            queryEmbedDim
            (BatchDim queryShape keyShape valueShape)
            (QuerySeqDim queryShape)
            (TransposeAndReshape headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim queryShape keyShape valueShape attentionMaskShape (BatchDim queryShape keyShape valueShape) (QuerySeqDim queryShape) (KeySeqDim keyShape valueShape))
        )
  type
    ForwardGeneratorOutput
      (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      Generator (device <+> queryDevice <+> keyDevice <+> generatorDevice)
  forward mha@MultiHeadAttention {..} (query, key, value, attentionMask) =
    let batchDim = case dimVal @(BatchDim queryShape keyShape valueShape) of
          Dim (Name name) (Size size) -> Dim name size
          Dim _ _ -> unsafeGetBatchDim (shape query) (shape key) (shape value)
        querySeqDim = case dimVal @(QuerySeqDim queryShape) of
          Dim (Name name) (Size size) -> Dim name size
          Dim _ _ -> unsafeGetQuerySeqDim (shape query)
        keySeqDim = case dimVal @(KeySeqDim keyShape valueShape) of
          Dim (Name name) (Size size) -> Dim name size
          Dim _ _ -> unsafeGetKeySeqDim (shape key) (shape value)
        embedDim = case dimVal @embedDim of
          Dim (Name name) (Size size) -> Dim name size
          Dim _ _ -> unsafeGetEmbedDim mha
        scaling :: Double = sqrt . fromIntegral . dimSize $ mhaHeadDim
     in runIxState $
          let q =
                ireturn query
                  >>>= IxState . forward mhaQInProj
                  >>>= ireturn . flip divScalar scaling
                  >>>= ireturn . reshape' @(BatchDim queryShape keyShape valueShape) @(QuerySeqDim queryShape) @headDim @headEmbedDim [batchDim, querySeqDim, mhaHeadDim, mhaHeadEmbedDim]
                  >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              k =
                ireturn key
                  >>>= IxState . forward mhaKInProj
                  >>>= ireturn . reshape' @(BatchDim queryShape keyShape valueShape) @(KeySeqDim keyShape valueShape) @headDim @headEmbedDim [batchDim, keySeqDim, mhaHeadDim, mhaHeadEmbedDim]
                  >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              kt = k >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3))
              weights =
                matmul <<$>> q <<*>> kt
                  >>>= IxState . forward mhaDropout . softmax @( 'SelectDim ( 'ByIndex 3))
                  >>>= ireturn . (`add` unsqueeze @( 'SelectDim ( 'ByIndex 1)) attentionMask)
              v =
                ireturn value
                  >>>= IxState . forward mhaVInProj
                  >>>= ireturn . reshape' @(BatchDim queryShape keyShape valueShape) @(KeySeqDim keyShape valueShape) @headDim @headEmbedDim [batchDim, keySeqDim, mhaHeadDim, mhaHeadEmbedDim]
                  >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
           in matmul <<$>> weights <<*>> v
                >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                >>>= ireturn . reshape'' @(BatchDim queryShape keyShape valueShape) @(QuerySeqDim queryShape) @embedDim [batchDim, querySeqDim, embedDim]
                >>>= IxState . forward mhaOutProj
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
