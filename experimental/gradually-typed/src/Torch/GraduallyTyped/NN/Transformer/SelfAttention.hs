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
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map as Map
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention, MultiHeadAttentionSpec (MultiHeadAttentionSpec))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor, TensorSpec (..))
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
  where
  SelfAttention ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim.
    GSelfAttention
      (SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
      (SALayerNormF style gradient device dataType queryEmbedDim)
      (SADropoutF style) ->
    SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim

data
  SelfAttentionSpec
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  where
  SelfAttentionSpec ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim.
    STransformerStyle style ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim queryEmbedDim ->
    Double ->
    Double ->
    SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim

type instance ModelSpec (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim) = SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim

type family
  SAMultiheadAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim =
    MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim

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
    (style :: TransformerStyle) ::
    Type
  where
  SADropoutF _ = Dropout

instance
  ( multiHeadAttention ~ SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim,
    HasInitialize multiHeadAttention device multiHeadAttention device,
    layerNorm ~ SALayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    dropout ~ SADropoutF style,
    HasInitialize dropout device dropout device
  ) =>
  HasInitialize
    (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    generatorDevice
    (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    device
  where
  initialize (SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        multiHeadAttention = IxStateT . initialize @multiHeadAttention $ MultiHeadAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $ case style of
          ST5 -> layerNormWithoutBiasSpec
          SByT5 -> layerNormWithoutBiasSpec
          SBART -> layerNormWithBiasSpec
          SMBART -> layerNormWithBiasSpec
          SPegasus -> layerNormWithBiasSpec
          SBERT -> layerNormWithBiasSpec
          SRoBERTa -> layerNormWithBiasSpec
          SGPT2 -> undefined
        dropout = IxStateT . initialize @dropout $ Dropout dropoutP
     in runIxStateT
          ( (GSelfAttention <<$>> multiHeadAttention <<*>> layerNorm <<*>> dropout)
              >>>= ireturn . SelfAttention
          )
          generator'

instance
  SingI style =>
  HasStateDict
    (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
  where
  fromStateDict (SelfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps) k =
    let multiHeadAttention ST5 = fromStateDict (MultiHeadAttentionSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) (k <> "SelfAttention.")
        multiHeadAttention SByT5 = fromStateDict (MultiHeadAttentionSpec SByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) (k <> "SelfAttention.")
        multiHeadAttention SBART = fromStateDict (MultiHeadAttentionSpec SBART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) (k <> "self_attn.")
        multiHeadAttention SMBART = fromStateDict (MultiHeadAttentionSpec SMBART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) (k <> "self_attn.")
        multiHeadAttention SPegasus = fromStateDict (MultiHeadAttentionSpec SPegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) (k <> "self_attn.")
        multiHeadAttention SBERT = fromStateDict (MultiHeadAttentionSpec SBERT gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) k
        multiHeadAttention SRoBERTa = fromStateDict (MultiHeadAttentionSpec SRoBERTa gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP) k
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = fromStateDict (LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict (LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "layer_norm.")
        layerNorm SBART = fromStateDict (LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "self_attn_layer_norm.")
        layerNorm SMBART = fromStateDict (LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "self_attn_layer_norm.")
        layerNorm SPegasus = fromStateDict (LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "self_attn_layer_norm.")
        layerNorm SBERT = fromStateDict (LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "output.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict (LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps) (k <> "output.LayerNorm.")
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict (Dropout dropoutP) k
     in SelfAttention
          <$> ( GSelfAttention
                  <$> multiHeadAttention style
                  <*> layerNorm style
                  <*> dropout style
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
    HasForward
      (SALayerNormF 'T5 gradient device dataType queryEmbedDim)
      query
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (SAMultiheadAttentionF 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generatorDevice
      mhaOutput
      device2,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutputDevice ~ (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)

testSA :: IO _
testSA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      generatorDevice = SUncheckedDevice CPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      dropoutP :: Double = 0.0
      eps :: Double = 1e-6
  let g = sMkGenerator generatorDevice 0
  (sa, g') <-
    initialize
      (SelfAttentionSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps)
      g
  sa' <- flip evalStateT Map.empty $ do
    toStateDict "sa." sa
    fromStateDict
      (SelfAttentionSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps)
      "sa."
  let batchDim = SName @"*" :&: SSize @1
      seqDim = SName @"*" :&: SSize @4
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward sa' (query, attentionBias) g'
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
    HasForward
      (SALayerNormF 'ByT5 gradient device dataType queryEmbedDim)
      query
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (SAMultiheadAttentionF 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
      (layerNormOutput, layerNormOutput, layerNormOutput, attentionBias)
      generatorDevice
      mhaOutput
      device2,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutputDevice ~ (device1 <+> device2)
  ) =>
  HasForward
    (SelfAttention 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim)
      (query, query, query, attentionBias)
      generatorDevice
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice),
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
    generatorOutputDevice ~ (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BERT gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim)
      (query, query, query, attentionBias)
      generatorDevice
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice),
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
    generatorOutputDevice ~ (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'BERT gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'RoBERTa gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim)
      (query, query, query, attentionBias)
      generatorDevice
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice),
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
    generatorOutputDevice ~ (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'RoBERTa gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
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
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryGradient ~ (gradient <|> queryGradient),
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithBiasF ('Shape '[queryEmbedDim]) ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor normedQueryGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim)
      (normedQuery, normedQuery, normedQuery, attentionBias)
      generatorDevice
      ( Tensor
          (gradient <|> queryGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice),
    output
      ~ Tensor
          (queryGradient <|> gradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutputDevice ~ (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (SelfAttention GSelfAttention {..}) (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
