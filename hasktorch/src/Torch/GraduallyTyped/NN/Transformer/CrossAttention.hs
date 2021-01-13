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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.CrossAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormC, LayerNorm)
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (HasInitializeMultiHeadAttentionC, MultiHeadAttention, MultiHeadAttentionOutputShape)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor (TransposeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, UnsqueezeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  CrossAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  CrossAttention ::
    forall device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP.
    { -- | cross-attention
      caMultiheadAttention :: MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP,
      -- | layer norm
      caLayerNorm :: LayerNorm device dataType ( 'Shape '[queryEmbedDim]),
      -- | dropout
      caDropout :: Dropout dropoutP
    } ->
    CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP

type HasInitializeCrossAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP =
  ( HasInitializeMultiHeadAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP,
    HasInitializeLayerNormC device dataType ( 'Shape '[queryEmbedDim]),
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))),
    WithDimC keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))
  )

instance
  HasInitializeCrossAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP =>
  HasInitialize (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
  where
  type
    InitializeF (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP) =
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
                                (dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))
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
                            withDim @keyEmbedDim @(dropoutP -> Double -> Generator device -> (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)) $
                              \keyEmbedDim ->
                                go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps = runState $ do
        multiheadAttention <-
          state $
            withoutDim @keyEmbedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP)
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
                      queryEmbedDim
                  )
                  keyEmbedDim
              )
              keyEmbedDim
              dropoutP
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
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ CrossAttention multiheadAttention layerNorm dropout

type CrossAttentionTransposeAndReshape
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat))
  (transposed :: Shape [Dim (Name Symbol) (Size Nat)])
  (transposed' :: Shape [Dim (Name Symbol) (Size Nat)]) =
  ReshapeF
    ( TransposeF
        ( 'SelectDim ( 'ByIndex 1))
        ( 'SelectDim ( 'ByIndex 2))
        ( MatmulF
            ( BroadcastShapesF
                ( SoftmaxF
                    ( 'SelectDim ( 'ByIndex 3))
                    ( MatmulF
                        transposed
                        ( TransposeF
                            ( 'SelectDim ( 'ByIndex 2))
                            ( 'SelectDim ( 'ByIndex 3))
                            transposed'
                        )
                    )
                )
                ( UnsqueezeF
                    ( 'SelectDim ( 'ByIndex 1))
                    attentionMaskShape
                )
            )
            transposed'
        )
    )
    ( 'Shape '[normedBatchDim, normedQuerySeqDim, embedDim])

type CrossAttentionTransposeAndReshape'
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat))
  (keySeqDim :: Dim (Name Symbol) (Size Nat)) =
  CrossAttentionTransposeAndReshape
    embedDim    attentionMaskShape
    normedBatchDim
    normedQuerySeqDim
    ( TransposeF
        ( 'SelectDim ( 'ByIndex 1))
        ( 'SelectDim ( 'ByIndex 2))
        ( ReshapeF
            ( LinearWithBiasF
                ( 'Shape '[embedDim, queryEmbedDim])
                ( 'Shape '[embedDim])
                normedQueryShape
            )
            ( 'Shape
                '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim]
            )
        )
    )
    ( TransposeF
        ( 'SelectDim ( 'ByIndex 1))
        ( 'SelectDim ( 'ByIndex 2))
        ( ReshapeF
            ( LinearWithBiasF
                ( 'Shape '[embedDim, keyEmbedDim])
                ( 'Shape '[embedDim])
                keyShape
            )
            ( 'Shape
                '[normedBatchDim, keySeqDim, headDim, headEmbedDim]
            )
        )
    )

type CrossAttentionTransposeAndReshape''
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  CrossAttentionTransposeAndReshape'
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    keyEmbedDim
    normedQueryShape
    keyShape
    attentionMaskShape
    (normedQueryShape ! 0 <+> keyShape ! 0)
    (normedQueryShape ! 1)
    (keyShape ! 1)

type CrossAttentionOutputShape
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  BroadcastShapesF
    queryShape
    ( LinearWithBiasF
        ( 'Shape '[queryEmbedDim, embedDim])
        ( 'Shape '[queryEmbedDim])
        ( CrossAttentionTransposeAndReshape''
            headDim
            headEmbedDim
            embedDim
            queryEmbedDim
            keyEmbedDim
            (LayerNormF ( 'Shape '[queryEmbedDim]) ( 'Shape '[queryEmbedDim]) queryShape)
            keyShape
            attentionMaskShape
        )
    )

instance
  ( KnownDim queryEmbedDim,
    HasForward
      (MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP)
      ( Tensor requiresGradient ( 'Layout 'Dense <+> queryLayout) (device <+> queryDevice) (dataType <+> queryDataType) (LayerNormF ( 'Shape '[queryEmbedDim]) ( 'Shape '[queryEmbedDim]) queryShape),
        Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice),
    Scalar dropoutP
  ) =>
  HasForward
    (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      Tensor
        requiresGradient
        (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionMaskLayout)
        (queryDevice <+> device <+> keyDevice <+> generatorDevice <+> attentionMaskDevice)
        (queryDataType <+> dataType <+> keyDataType <+> attentionMaskDataType)
        (CrossAttentionOutputShape headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim queryShape keyShape attentionMaskShape)
  type
    ForwardGeneratorOutput
      (CrossAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      Generator (device <+> queryDevice <+> keyDevice <+> generatorDevice <+> attentionMaskDevice)
  forward CrossAttention {..} (query, key, attentionMask) =
    runIxState $
      ireturn query
        >>>= IxState . forward caLayerNorm
        >>>= (\query' -> IxState $ forward caMultiheadAttention (query', key, key, attentionMask))
        >>>= IxState . forward caDropout
        >>>= ireturn . (query `add`)