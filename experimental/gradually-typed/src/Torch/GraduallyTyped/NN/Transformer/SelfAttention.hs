{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TupleSections #-}
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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL9
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL9C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.SelfAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention, lookupMultiHeadAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SDim, SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, SGetDim, SGetShape)
import Torch.GraduallyTyped.Unify (type (<+>))

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
  SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
    MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP

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
  SALayerNormF 'BERT device dataType queryEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'RoBERTa device dataType queryEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'BART device dataType queryEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'Pegasus device dataType queryEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])

type family
  SADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  SADropoutF _ dropoutP = Dropout dropoutP

instance
  ( Scalar dropoutP,
    multiHeadAttention ~ SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitialize multiHeadAttention,
    layerNorm ~ SALayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ (SDevice device -> SDataType dataType -> SShape ('Shape '[queryEmbedDim]) -> Double -> layerNorm),
    dropout ~ SADropoutF style dropoutP,
    HasInitialize dropout
  ) =>
  HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim queryEmbedDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps = runState $ do
    multiHeadAttention <-
      state $ initialize @multiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP
    let layerNorm = initialize @layerNorm device dataType (SShape $ queryEmbedDim :|: SNil) eps
    let dropout = initialize @dropout dropoutP
    pure . SelfAttention $ GSelfAttention multiHeadAttention layerNorm dropout

lookupSelfAttention ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    Scalar dropoutP
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  Double ->
  String ->
  m (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix =
  let selfAttention ST5 = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "SelfAttention.")
      selfAttention SBERT = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP prefix
      selfAttention SRoBERTa = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP prefix
      selfAttention SBART = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "self_attn.")
      selfAttention SPegasus = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "self_attn.")
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      layerNorm SBERT =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "output.LayerNorm.weight")
          <*> lookupTensor (prefix <> "output.LayerNorm.bias")
          <*> pure eps
      layerNorm SRoBERTa =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "output.LayerNorm.weight")
          <*> lookupTensor (prefix <> "output.LayerNorm.bias")
          <*> pure eps
      layerNorm SBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "self_attn_layer_norm.weight")
          <*> lookupTensor (prefix <> "self_attn_layer_norm.bias")
          <*> pure eps
      layerNorm SPegasus =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "self_attn_layer_norm.weight")
          <*> lookupTensor (prefix <> "self_attn_layer_norm.bias")
          <*> pure eps
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
   in SelfAttention
        <$> ( GSelfAttention
                <$> selfAttention (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
            )

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
  ( SGetDim queryEmbedDim,
    SGetShape queryShape,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithoutBiasF ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (normedQuery, normedQuery, normedQuery, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward saLayerNorm
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
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
--         └─►saMultiheadAttention  │
--                      │           │
--                      ▼           │
--                  saDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                       saLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BART device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              (BroadcastShapesF queryShape mhaOutputShape)
          ),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'BART device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward saLayerNorm

-- | 'HasForward' instance for @SelfAttention 'BERT@.
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
--         └─►saMultiheadAttention  │
--                      │           │
--                      ▼           │
--                  saDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                       saLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BERT device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              (BroadcastShapesF queryShape mhaOutputShape)
          ),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'BERT device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward saLayerNorm

-- | 'HasForward' instance for @SelfAttention 'RoBERTa@.
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
--         └─►saMultiheadAttention  │
--                      │           │
--                      ▼           │
--                  saDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                       saLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'RoBERTa device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              (BroadcastShapesF queryShape mhaOutputShape)
          ),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'RoBERTa device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward saLayerNorm

-- | 'HasForward' instance for @SelfAttention 'Pegasus@.
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
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithBiasF ('Shape '[queryEmbedDim]) ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'Pegasus device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (normedQuery, normedQuery, normedQuery, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'Pegasus device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward saLayerNorm
        >>>= (\query' -> IxState $ forward saMultiheadAttention (query', query', query', attentionBias))
        >>>= IxState . forward saDropout
        >>>= ireturn . (query `add`)
