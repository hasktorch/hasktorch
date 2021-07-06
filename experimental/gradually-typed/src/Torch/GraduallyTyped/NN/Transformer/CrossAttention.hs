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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.CrossAttention where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, SDType (..), SDataType (..))
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

-- | Generic cross-attention layer.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'CrossAttention'.
data
  GCrossAttention
    (mha :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GCrossAttention ::
    forall mha layerNorm dropout.
    { -- | cross-attention
      caMultiHeadAttention :: mha,
      -- | layer norm
      caLayerNorm :: layerNorm,
      -- | dropout
      caDropout :: dropout
    } ->
    GCrossAttention mha layerNorm dropout

-- | Cross-attention layer.
newtype
  CrossAttention
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
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
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP.
    GCrossAttention
      (CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      (CALayerNormF style gradient device dataType queryEmbedDim)
      (CADropoutF style dropoutP) ->
    CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP

type family
  CAMultiheadAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP =
    MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP

type family
  CALayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CALayerNormF 'T5 gradient device dataType queryEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim])
  CALayerNormF 'ByT5 gradient device dataType queryEmbedDim = CALayerNormF 'T5 gradient device dataType queryEmbedDim
  CALayerNormF 'BART gradient device dataType queryEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim])
  CALayerNormF 'MBART gradient device dataType queryEmbedDim = CALayerNormF 'BART gradient device dataType queryEmbedDim
  CALayerNormF 'Pegasus gradient device dataType queryEmbedDim = CALayerNormF 'BART gradient device dataType queryEmbedDim

type family
  CADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  CADropoutF _ dropoutP =
    Dropout dropoutP

instance
  ( Scalar dropoutP,
    multiHeadAttention ~ CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
    HasInitialize multiHeadAttention (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, SDim keyEmbedDim, dropoutP) generator generator',
    layerNorm ~ CALayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[queryEmbedDim]), Double) generator' generator',
    dropout ~ CADropoutF style dropoutP
  ) =>
  HasInitialize
    (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, dropoutP, Double)
    generator
    generator'
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) =
    let multiHeadAttention = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP)
        layerNorm = IxState . initialize $ (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps)
        dropout = IxState . initialize $ dropoutP
     in runIxState $
          (GCrossAttention <<$>> multiHeadAttention <<*>> layerNorm <<*>> dropout)
            >>>= (ireturn . CrossAttention)

instance
  SingI style =>
  HasStateDict
    (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) k =
    let multiHeadAttention ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP) (k <> "EncDecAttention.")
        multiHeadAttention SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP) (k <> "EncDecAttention.")
        multiHeadAttention SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP) (k <> "encoder_attn.")
        multiHeadAttention SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP) (k <> "encoder_attn.")
        multiHeadAttention SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, keyEmbedDim, dropoutP) (k <> "encoder_attn.")
        multiHeadAttention SBERT = undefined
        multiHeadAttention SRoBERTa = undefined
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "encoder_attn_layer_norm.")
        layerNorm SMBART = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "encoder_attn_layer_norm.")
        layerNorm SPegasus = fromStateDict (gradient, device, dataType, SShape $ queryEmbedDim :|: SNil, eps) (k <> "encoder_attn_layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
     in CrossAttention
          <$> ( GCrossAttention
                  <$> multiHeadAttention (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
              )
  toStateDict k (CrossAttention GCrossAttention {..}) =
    let multiHeadAttention ST5 = toStateDict (k <> "EncDecAttention.")
        multiHeadAttention SByT5 = toStateDict (k <> "EncDecAttention.")
        multiHeadAttention SBART = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SMBART = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SPegasus = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SBERT = undefined
        multiHeadAttention SRoBERTa = undefined
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "layer_norm.")
        layerNorm SBART = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SMBART = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SPegasus = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = toStateDict k
     in do
          () <- multiHeadAttention (sing @style) caMultiHeadAttention
          () <- layerNorm (sing @style) caLayerNorm
          () <- dropout (sing @style) caDropout
          pure ()

-- | 'HasForward' instance for @CrossAttention 'T5@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    HasForward
      (CALayerNormF 'T5 gradient device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (CAMultiheadAttentionF 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      (layerNormOutput, key, key, attentionBias)
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
    (CrossAttention 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (query, key, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)

testCA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (sa, g') = initialize @(CrossAttention 'T5 _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @4
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward sa (query, key, attentionBias) g'
  pure output

-- | 'HasForward' instance for @CrossAttention 'ByT5@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    HasForward
      (CALayerNormF 'ByT5 gradient device dataType queryEmbedDim)
      query
      generator
      layerNormOutput
      generator,
    HasForward
      (CAMultiheadAttentionF 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      (layerNormOutput, key, key, attentionBias)
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
    (CrossAttention 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (query, key, attentionBias)
    generator
    output
    generatorOutput
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @CrossAttenton 'BART@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--   caLayerNorm
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    key ~ Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP)
      (query, key, key, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          (gradient <|> queryGradient <|> keyGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (gradient <|> queryGradient <|> keyGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              (BroadcastShapesF queryShape mhaOutputShape)
          ),
    generatorOutput ~ Generator (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (CrossAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (query, key, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward (caLayerNorm ca)

-- | 'HasForward' instance for @CrossAttention 'Pegasus@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    key ~ Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryGradient ~ (gradient <|> queryGradient),
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithBiasF ('Shape '[queryEmbedDim]) ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor normedQueryGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP)
      ( normedQuery,
        key,
        key,
        attentionBias
      )
      (Generator generatorDevice)
      ( Tensor
          (queryGradient <|> gradient <|> keyGradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          (queryGradient <|> gradient <|> keyGradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (CrossAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (query, key, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)
