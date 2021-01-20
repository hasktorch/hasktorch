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
{-# OPTIONS_GHC -v2
                -fomit-interface-pragmas
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

module Torch.GraduallyTyped.NN.Transformer.SelfAttention where

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
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithoutBiasF, LayerNormWithoutBiasSelectDimsF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm)
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (HasInitializeMultiHeadAttentionC, MultiHeadAttention)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithSelectDimsC, WithShapeC (..))
import Torch.GraduallyTyped.Tensor (ReshapeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  SelfAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SelfAttention ::
    forall device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP.
    { -- | self-attention
      saMultiheadAttention :: MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
      -- | layer norm
      saLayerNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]),
      -- | dropout
      saDropout :: Dropout dropoutP
    } ->
    SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP

type HasInitializeSelfAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
  ( HasInitializeMultiHeadAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    HasInitializeLayerNormWithoutBiasC device dataType ( 'Shape '[queryEmbedDim]),
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))),
    WithDimC embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))),
    WithDimC queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
  )

instance
  HasInitializeSelfAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =>
  HasInitialize (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) =
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
                            (dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
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
                        withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)) $
                          \queryEmbedDim ->
                            go deviceType dType headDim headEmbedDim embedDim queryEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps = runState $ do
        multiheadAttention <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(MultiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
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
                  queryEmbedDim
              )
              queryEmbedDim
              dropoutP
        let layerNorm =
              withoutShape @( 'Shape '[queryEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ SelfAttention multiheadAttention layerNorm dropout

type SelfAttentionTransposeAndReshape
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat))
  (transposed :: Shape [Dim (Name Symbol) (Size Nat)]) =
  TransposeF
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
                        transposed
                    )
                )
            )
            ( UnsqueezeF
                ( 'SelectDim ( 'ByIndex 1))
                attentionMaskShape
            )
        )
        transposed
    )

type SelfAttentionTransposeAndReshape'
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat)) =
  SelfAttentionTransposeAndReshape
    embedDim
    attentionMaskShape
    normedBatchDim
    normedQuerySeqDim
    ( TransposeF
        ( 'SelectDim ( 'ByIndex 1))
        ( 'SelectDim ( 'ByIndex 2))
        ( ReshapeF
            ( LinearWithoutBiasF
                ( 'Shape '[embedDim, queryEmbedDim])
                normedQueryShape
            )
            ( 'Shape
                '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim]
            )
        )
    )

type SelfAttentionTransposeAndReshape''
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat)) =
  ReshapeF
    ( SelfAttentionTransposeAndReshape'
        headDim
        headEmbedDim
        embedDim
        queryEmbedDim
        normedQueryShape
        attentionMaskShape
        normedBatchDim
        normedQuerySeqDim
    )
    ( 'Shape '[normedBatchDim, normedQuerySeqDim, embedDim])

type SelfAttentionTransposeAndReshape'''
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  SelfAttentionTransposeAndReshape''
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    normedQueryShape
    attentionMaskShape
    (normedQueryShape ! 0)
    (normedQueryShape ! 1)

type SelfAttentionOutputShape
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  BroadcastShapesF
    queryShape
    ( LinearWithoutBiasF
        ( 'Shape '[queryEmbedDim, embedDim])
        ( SelfAttentionTransposeAndReshape'''
            headDim
            headEmbedDim
            embedDim
            queryEmbedDim
            ( LayerNormWithoutBiasF
                ( 'Shape '[queryEmbedDim])
                queryShape
            )
            attentionMaskShape
        )
    )

instance
  ( KnownShape queryShape,
    normedQueryShape ~ (LayerNormWithoutBiasF ( 'Shape '[queryEmbedDim]) queryShape),
    KnownShape normedQueryShape,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    normedBatchDim ~ (normedQueryShape ! 0),
    KnownDim normedBatchDim,
    normedQuerySeqDim ~ (normedQueryShape ! 1),
    KnownDim normedQuerySeqDim,
    WithShapeC
      ( 'Shape '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( LinearWithoutBiasF
              ( 'Shape '[embedDim, queryEmbedDim])
              normedQueryShape
          ) ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( ReshapeF
              ( LinearWithoutBiasF
                  ( 'Shape '[embedDim, queryEmbedDim])
                  normedQueryShape
              )
              ( 'Shape '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim])
          )
      ),
    transposedAndReshaped
      ~ SelfAttentionTransposeAndReshape'
          headDim
          headEmbedDim
          embedDim
          queryEmbedDim
          normedQueryShape
          attentionMaskShape
          normedBatchDim
          normedQuerySeqDim,
    WithShapeC
      ( 'Shape '[normedBatchDim, normedQuerySeqDim, embedDim])
      ( Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout <+> attentionMaskLayout)
          (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice)
          (dataType <+> queryDataType <+> attentionMaskDataType)
          transposedAndReshaped ->
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> queryLayout <+> attentionMaskLayout)
          (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice)
          (dataType <+> queryDataType <+> attentionMaskDataType)
          ( ReshapeF
              transposedAndReshaped
              ( 'Shape '[normedBatchDim, normedQuerySeqDim, embedDim])
          )
      ),
    Scalar dropoutP
  ) =>
  HasForward
    (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      Tensor
        'WithGradient
        (queryLayout <+> 'Layout 'Dense <+> attentionMaskLayout)
        (queryDevice <+> device <+> generatorDevice <+> attentionMaskDevice)
        (queryDataType <+> dataType <+> attentionMaskDataType)
        (SelfAttentionOutputShape headDim headEmbedDim embedDim queryEmbedDim queryShape attentionMaskShape)
  type
    ForwardGeneratorOutput
      (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      (Generator (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice))
  forward SelfAttention {..} (query, attentionMask) =
    runIxState $
      ireturn query
        >>>= IxState . forward saLayerNorm
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionMask))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)