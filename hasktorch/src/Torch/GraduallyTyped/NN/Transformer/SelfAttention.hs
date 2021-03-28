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
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm)
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (HasInitializeMultiHeadAttentionC, MultiHeadAttention)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor (ReshapeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | T5-style self-attention layer without biases.
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
      saLayerNorm :: LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim]),
      -- | dropout
      saDropout :: Dropout dropoutP
    } ->
    SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP

type HasInitializeSelfAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
  ( HasInitializeMultiHeadAttentionC device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    HasInitializeLayerNormWithoutBiasC device dataType ('Shape '[queryEmbedDim]),
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
              withoutShape @('Shape '[queryEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim]))
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
  (attentionBiasShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat))
  (transposed :: Shape [Dim (Name Symbol) (Size Nat)]) =
  TransposeF
    ('SelectDim ('ByIndex 1))
    ('SelectDim ('ByIndex 2))
    ( MatmulF
        ( SoftmaxF
            ('SelectDim ('ByIndex 3))
            ( BroadcastShapesF
                ( MatmulF
                    transposed
                    ( TransposeF
                        ('SelectDim ('ByIndex 2))
                        ('SelectDim ('ByIndex 3))
                        transposed
                    )
                )
                attentionBiasShape
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
  (attentionBiasShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat)) =
  SelfAttentionTransposeAndReshape
    embedDim
    attentionBiasShape
    normedBatchDim
    normedQuerySeqDim
    ( TransposeF
        ('SelectDim ('ByIndex 1))
        ('SelectDim ('ByIndex 2))
        ( ReshapeF
            ( LinearWithoutBiasF
                ('Shape '[embedDim, queryEmbedDim])
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
  (attentionBiasShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (normedBatchDim :: Dim (Name Symbol) (Size Nat))
  (normedQuerySeqDim :: Dim (Name Symbol) (Size Nat)) =
  ReshapeF
    ( SelfAttentionTransposeAndReshape'
        headDim
        headEmbedDim
        embedDim
        queryEmbedDim
        normedQueryShape
        attentionBiasShape
        normedBatchDim
        normedQuerySeqDim
    )
    ('Shape '[normedBatchDim, normedQuerySeqDim, embedDim])

type SelfAttentionTransposeAndReshape'''
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (normedQueryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionBiasShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  SelfAttentionTransposeAndReshape''
    headDim
    headEmbedDim
    embedDim
    queryEmbedDim
    normedQueryShape
    attentionBiasShape
    (normedQueryShape ! 0)
    (normedQueryShape ! 1)

type SelfAttentionOutputShape
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionBiasShape :: Shape [Dim (Name Symbol) (Size Nat)]) =
  BroadcastShapesF
    queryShape
    ( LinearWithoutBiasF
        ('Shape '[queryEmbedDim, embedDim])
        ( SelfAttentionTransposeAndReshape'''
            headDim
            headEmbedDim
            embedDim
            queryEmbedDim
            ( LayerNormWithoutBiasF
                ('Shape '[queryEmbedDim])
                queryShape
            )
            attentionBiasShape
        )
    )

-- | 'HasForward' instance for 'SelfAttention'.
--
-- @
-- ┌───────────────┐     ┌───────┐
-- │ attentionBias │     │ query │
-- └───────┬───────┘     └───┬───┘
--         │                 │
--         │           ┌─────┴─────┐
--         │           │           │
--         │           ▼           │
--         │      saLayerNorm      │
--         │           │           │
--         │      ┌────┼────┐      │
--         │      │    │    │      │
--         │      ▼    ▼    ▼      │
--         └─►saMultiheadAttention │
--                     │           │
--                     ▼           │
--                 saDropout       │
--                     │           │
--                     └───►add◄───┘
--                           │
--                           ▼
--                       ┌───────┐
--                       │ query │
--                       └───────┘
-- @
instance
  ( KnownShape queryShape,
    normedQueryShape ~ LayerNormWithoutBiasF ('Shape '[queryEmbedDim]) queryShape,
    KnownShape normedQueryShape,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    normedBatchDim ~ (normedQueryShape ! 0),
    KnownDim normedBatchDim,
    normedQuerySeqDim ~ (normedQueryShape ! 1),
    KnownDim normedQuerySeqDim,
    WithShapeC
      ('Shape '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim])
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( LinearWithoutBiasF
              ('Shape '[embedDim, queryEmbedDim])
              normedQueryShape
          ) ->
        Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice)
          (dataType <+> queryDataType)
          ( ReshapeF
              ( LinearWithoutBiasF
                  ('Shape '[embedDim, queryEmbedDim])
                  normedQueryShape
              )
              ('Shape '[normedBatchDim, normedQuerySeqDim, headDim, headEmbedDim])
          )
      ),
    transposedAndReshaped
      ~ SelfAttentionTransposeAndReshape'
          headDim
          headEmbedDim
          embedDim
          queryEmbedDim
          normedQueryShape
          attentionBiasShape
          normedBatchDim
          normedQuerySeqDim,
    WithShapeC
      ('Shape '[normedBatchDim, normedQuerySeqDim, embedDim])
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          transposedAndReshaped ->
        Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          ( ReshapeF
              transposedAndReshaped
              ('Shape '[normedBatchDim, normedQuerySeqDim, embedDim])
          )
      ),
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (SelfAttentionOutputShape headDim headEmbedDim embedDim queryEmbedDim queryShape attentionBiasShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward SelfAttention {..} (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward saLayerNorm
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for 'BARTSelfAttention'.
--
-- @
-- ┌───────────────┐      ┌───────┐
-- │ attentionBias │      │ query │
-- └───────┬───────┘      └───┬───┘
--         │                  │
--         │            ┌─────┴─────┐
--         │            │           │
--         │       ┌────┼────┐      │
--         │       │    │    │      │
--         │       ▼    ▼    ▼      │
--         └─►bsaMultiheadAttention │
--                      │           │
--                      ▼           │
--                 bsaDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                      bsaLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @