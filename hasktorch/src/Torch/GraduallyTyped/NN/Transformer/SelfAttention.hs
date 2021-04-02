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
import Data.Kind (Constraint, Type)
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
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor (ReshapeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (Unify, type (<+>), type (<|>))

-- | Generic self-attention layer.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'SelfAttention'.
data
  GSelfAttention
    (mha :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GSelfAttention ::
    forall mha layerNorm dropout.
    { -- | self-attention
      saMultiheadAttention :: mha,
      -- | layer norm
      saLayerNorm :: layerNorm,
      -- | dropout
      saDropout :: dropout
    } ->
    GSelfAttention mha layerNorm dropout

-- | Self-attention layer.
newtype
  SelfAttention
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SelfAttention ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP.
    GSelfAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP ->
    SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP

type GSelfAttentionF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GSelfAttention
    (SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (SALayerNormF style device dataType queryEmbedDim)
    (SADropoutF style dropoutP)

type family
  SAMultiheadAttentionF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  SAMultiheadAttentionF 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
    MultiHeadAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP

type family
  SALayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SALayerNormF 'T5 device dataType queryEmbedDim =
    LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])

type family
  SADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  SADropoutF 'T5 dropoutP =
    Dropout dropoutP

type HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))),
    WithDimC embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))),
    WithDimC queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
  )

instance
  ( HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    Scalar dropoutP,
    multiHeadAttention ~ SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitialize multiHeadAttention,
    HasInitializeMultiHeadAttentionC multiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    InitializeF multiHeadAttention ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF queryEmbedDim (WithDimF queryEmbedDim (dropoutP -> Generator device -> (multiHeadAttention, Generator device))))))))),
    layerNorm ~ SALayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    dropout ~ SADropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout),
    WithDimsC '[queryEmbedDim] (Double -> layerNorm),
    WithDataTypeC dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm)),
    WithDeviceC device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm)))
  ) =>
  HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) =
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
                            (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
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
                        withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)) $
                          \queryEmbedDim ->
                            go deviceType dType headDim headEmbedDim embedDim queryEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps = runState $ do
        multiHeadAttention <-
          state $
            withoutDim @queryEmbedDim @(dropoutP -> Generator device -> (multiHeadAttention, Generator device))
              ( withoutDim @queryEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @multiHeadAttention
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
              withoutShape @('Shape '[queryEmbedDim]) @(Double -> layerNorm)
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @layerNorm
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
        let dropout = initialize @dropout dropoutP
        pure . SelfAttention $ GSelfAttention multiHeadAttention layerNorm dropout

-- | 'HasForward' instance for @SelfAttention 'T5@.
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
  ( KnownDim queryEmbedDim,
    KnownShape queryShape,
    Scalar dropoutP,
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithoutBiasF ('Shape '[queryEmbedDim]) queryShape,
    HasForward
      (MultiHeadAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      ( Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention sa) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward (saLayerNorm sa)
        >>>= (\query' -> IxState $ forward (saMultiheadAttention sa) (query', query', query', attentionBias))
        >>>= IxState . forward (saDropout sa)
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @SelfAttention 'BART@.
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