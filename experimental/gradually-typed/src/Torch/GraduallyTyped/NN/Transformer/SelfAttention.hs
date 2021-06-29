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
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention, lookupMultiHeadAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

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
  SALayerNormF 'T5 device dataType queryEmbedDim = LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'ByT5 device dataType queryEmbedDim = SALayerNormF 'T5 device dataType queryEmbedDim
  SALayerNormF 'BART device dataType queryEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'MBART device dataType queryEmbedDim = SALayerNormF 'BART device dataType queryEmbedDim
  SALayerNormF 'Pegasus device dataType queryEmbedDim = SALayerNormF 'BART device dataType queryEmbedDim
  SALayerNormF 'BERT device dataType queryEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'RoBERTa device dataType queryEmbedDim = SALayerNormF 'BERT device dataType queryEmbedDim

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
    HasInitialize multiHeadAttention (SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim queryEmbedDim, SDim queryEmbedDim, dropoutP) generator generator',
    layerNorm ~ SALayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm (SDevice device, SDataType dataType, SShape ('Shape '[queryEmbedDim]), Double) generator' generator',
    dropout ~ SADropoutF style dropoutP,
    HasInitialize dropout dropoutP generator' generator'
  ) =>
  HasInitialize
    (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, dropoutP, Double)
    generator
    generator'
  where
  initialize (device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) =
    let multiHeadAttention = IxState $ initialize (device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP)
        layerNorm = IxState $ initialize (device, dataType, SShape $ queryEmbedDim :|: SNil, eps)
        dropout = IxState $ initialize dropoutP
     in runIxState $
          (GSelfAttention <<$>> multiHeadAttention <<*>> layerNorm <<*>> dropout)
            >>>= ireturn . SelfAttention

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
      selfAttention SByT5 = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "SelfAttention.")
      selfAttention SBART = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "self_attn.")
      selfAttention SMBART = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "self_attn.")
      selfAttention SPegasus = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP (prefix <> "self_attn.")
      selfAttention SBERT = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP prefix
      selfAttention SRoBERTa = lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP prefix
      selfAttention SGPT2 = undefined
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      layerNorm SByT5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      layerNorm SBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "self_attn_layer_norm.weight")
          <*> lookupTensor (prefix <> "self_attn_layer_norm.bias")
          <*> pure eps
      layerNorm SMBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "self_attn_layer_norm.weight")
          <*> lookupTensor (prefix <> "self_attn_layer_norm.bias")
          <*> pure eps
      layerNorm SPegasus =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "self_attn_layer_norm.weight")
          <*> lookupTensor (prefix <> "self_attn_layer_norm.bias")
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
      layerNorm SGPT2 = undefined
      dropout _ = pure (Dropout dropoutP)
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
    Scalar dropoutP,
    HasForward
      (SALayerNormF 'T5 device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (SAMultiheadAttentionF 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generator
      mhaOutput
      mhaGeneratorOutput,
    query ~ Tensor requiresGradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor requiresGradient1 layout1 device1 dataType1 shape1,
    mhaGeneratorOutput ~ Generator device2,
    output
      ~ Tensor
          (requiresGradient0 <|> requiresGradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutput ~ Generator (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    generator
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

testSA = do
  let device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      dropoutP :: Float = 0.0
      eps :: Double = 1e-6
  g <- sMkGenerator device 0
  let (sa, g') =
        initialize
          @(SelfAttention 'T5 _ _ _ _ _ _ _)
          (device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps)
          g
      batchDim = SName @"*" :&: SSize @1
      seqDim = SName @"*" :&: SSize @4
      sOnes' = sOnes SWithoutGradient (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  let (output, _) = forward sa (query, attentionBias) g'
  pure output

-- | 'HasForward' instance for @SelfAttention 'ByT5@.
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
    HasForward
      (SALayerNormF 'ByT5 device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (SAMultiheadAttentionF 'ByT5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generator
      mhaOutput
      mhaGeneratorOutput,
    query ~ Tensor requiresGradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor requiresGradient1 layout1 device1 dataType1 shape1,
    mhaGeneratorOutput ~ Generator device2,
    output
      ~ Tensor
          (requiresGradient0 <|> requiresGradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutput ~ Generator (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'ByT5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    generator
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
