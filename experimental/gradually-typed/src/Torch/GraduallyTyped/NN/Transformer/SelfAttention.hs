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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL9C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.SelfAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
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
      saMultiHeadAttention :: mha,
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
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SelfAttention ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP.
    GSelfAttention
      (SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (SALayerNormF style gradient device dataType queryEmbedDim)
      (SADropoutF style dropoutP) ->
    SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP

type family
  SAMultiheadAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
    MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP

type family
  SALayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SALayerNormF 'T5 gradient device dataType queryEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'ByT5 gradient device dataType queryEmbedDim = SALayerNormF 'T5 gradient device dataType queryEmbedDim
  SALayerNormF 'BART gradient device dataType queryEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'MBART gradient device dataType queryEmbedDim = SALayerNormF 'BART gradient device dataType queryEmbedDim
  SALayerNormF 'Pegasus gradient device dataType queryEmbedDim = SALayerNormF 'BART gradient device dataType queryEmbedDim
  SALayerNormF 'BERT gradient device dataType queryEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'RoBERTa gradient device dataType queryEmbedDim = SALayerNormF 'BERT gradient device dataType queryEmbedDim

type family
  SADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  SADropoutF _ dropoutP = Dropout dropoutP

instance
  ( Scalar dropoutP,
    multiHeadAttention ~ SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitialize multiHeadAttention (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim queryEmbedDim, SDim queryEmbedDim, dropoutP) generator generator',
    layerNorm ~ SALayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[queryEmbedDim]), Double) generator' generator',
    dropout ~ SADropoutF style dropoutP,
    HasInitialize dropout dropoutP generator' generator'
  ) =>
  HasInitialize
    (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, dropoutP, Double)
    generator
    generator'
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) =
    let multiHeadAttention = IxState $ initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP)
        layerNorm = IxState $ initialize (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps)
        dropout = IxState $ initialize dropoutP
     in runIxState $
          (GSelfAttention <<$>> multiHeadAttention <<*>> layerNorm <<*>> dropout)
            >>>= ireturn . SelfAttention

instance
  SingI style =>
  HasStateDict
    (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k =
    let multiHeadAttention ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) (k <> "SelfAttention.")
        multiHeadAttention SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) (k <> "SelfAttention.")
        multiHeadAttention SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) (k <> "self_attn.")
        multiHeadAttention SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) (k <> "self_attn.")
        multiHeadAttention SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) (k <> "self_attn.")
        multiHeadAttention SBERT = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) k
        multiHeadAttention SRoBERTa = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, queryEmbedDim, queryEmbedDim, dropoutP) k
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "self_attn_layer_norm.")
        layerNorm SMBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "self_attn_layer_norm.")
        layerNorm SPegasus = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "self_attn_layer_norm.")
        layerNorm SBERT = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
     in SelfAttention
          <$> ( GSelfAttention
                  <$> multiHeadAttention (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
              )
  toStateDict k (SelfAttention GSelfAttention {..}) =
    let multiHeadAttention ST5 = toStateDict (k <> "SelfAttention.")
        multiHeadAttention SByT5 = toStateDict (k <> "SelfAttention.")
        multiHeadAttention SBART = toStateDict (k <> "self_attn.")
        multiHeadAttention SMBART = toStateDict (k <> "self_attn.")
        multiHeadAttention SPegasus = toStateDict (k <> "self_attn.")
        multiHeadAttention SBERT = toStateDict k
        multiHeadAttention SRoBERTa = toStateDict k
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "layer_norm.")
        layerNorm SBART = toStateDict (k <> "self_attn_layer_norm.")
        layerNorm SMBART = toStateDict (k <> "self_attn_layer_norm.")
        layerNorm SPegasus = toStateDict (k <> "self_attn_layer_norm.")
        layerNorm SBERT = toStateDict (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = toStateDict (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
        dropout _ = toStateDict k
     in do
          () <- multiHeadAttention (sing @style) saMultiHeadAttention
          () <- layerNorm (sing @style) saLayerNorm
          () <- dropout (sing @style) saDropout
          pure ()

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
--         └─►saMultiHeadAttention │
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
      (SALayerNormF 'T5 gradient device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (SAMultiheadAttentionF 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generator
      mhaOutput
      mhaGeneratorOutput,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    mhaGeneratorOutput ~ Generator device2,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutput ~ Generator (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)

testSA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
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
          @(SelfAttention 'T5 _ _ _ _ _ _ _ _)
          (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps)
          g
      batchDim = SName @"*" :&: SSize @1
      seqDim = SName @"*" :&: SSize @4
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward sa (query, attentionBias) g'
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
--         └─►saMultiHeadAttention │
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
      (SALayerNormF 'ByT5 gradient device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (SAMultiheadAttentionF 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generator
      mhaOutput
      mhaGeneratorOutput,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    mhaGeneratorOutput ~ Generator device2,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutput ~ Generator (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
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
--         └─►saMultiHeadAttention  │
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
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
    (SelfAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward saLayerNorm

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
--         └─►saMultiHeadAttention  │
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BERT gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
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
    (SelfAttention 'BERT gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward saLayerNorm

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
--         └─►saMultiHeadAttention  │
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'RoBERTa gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
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
    (SelfAttention 'RoBERTa gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward saLayerNorm

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
--         └─►saMultiHeadAttention │
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryGradient ~ (gradient <|> queryGradient),
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithBiasF ('Shape '[queryEmbedDim]) ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor normedQueryGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (normedQuery, normedQuery, normedQuery, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (queryGradient <|> gradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
