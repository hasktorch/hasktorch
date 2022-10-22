{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GCrossAttention where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttention (GMultiHeadAttentionF, multiHeadAttentionSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), HasDropout (..), SHasBias (..), SHasDropout (..))
import Torch.GraduallyTyped.Prelude (Catch, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SShape (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic cross-attention layer data type.
--
-- - @initialLayerNorm@: the initial layer normalization
-- - @mha@: the multi-headed attention layer
-- - @dropout@: the dropout layer
-- - @finalLayerNorm@: the final layer normalization
data
  GCrossAttention
    (initialLayerNorm :: Type)
    (mha :: Type)
    (dropout :: Type)
    (finalLayerNorm :: Type)
  where
  GCrossAttention ::
    forall initialLayerNorm mha dropout finalLayerNorm.
    { -- | initial layer normalization of the cross-attention layer.
      caInitialLayerNorm :: initialLayerNorm,
      -- | multi-headed attention layer specialized for cross-attention.
      caMultiHeadAttention :: mha,
      -- | dropout
      caDropout :: dropout,
      -- | final layer normalization of the cross-attention layer.
      caFinalLayerNorm :: finalLayerNorm
    } ->
    GCrossAttention initialLayerNorm mha dropout finalLayerNorm
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GCrossAttention initialLayerNorm mha dropout finalLayerNorm) =
    GCrossAttention (ModelSpec initialLayerNorm) (ModelSpec mha) (ModelSpec dropout) (ModelSpec finalLayerNorm)

type family
  GCrossAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (hasDropout :: HasDropout) ::
    Type
  where
  GCrossAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout =
    GCrossAttention
      (CAInitialLayerNormF style gradient device dataType queryEmbedDim)
      (CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout)
      (CADropoutF style hasDropout)
      (CAFinalLayerNormF style gradient device dataType queryEmbedDim)

-- | Specifies the initial layer normalization of the cross-attention layer.
type family
  CAInitialLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim]))
  CAInitialLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    CAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim
  CAInitialLayerNormF 'BART _ _ _ _ =
    ()
  CAInitialLayerNormF 'MBART gradient device dataType queryEmbedDim =
    CAInitialLayerNormF 'BART gradient device dataType queryEmbedDim
  CAInitialLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))

-- | Specifies the multi-headed attention layer specialized for cross-attention.
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
    (hasDropout :: HasDropout) ::
    Type
  where
  CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout =
    NamedModel (GMultiHeadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim hasDropout)

-- | Specifies the dropout layer of the cross-attention layer.
type family
  CADropoutF
    (style :: TransformerStyle)
    (hasDropout :: HasDropout) ::
    Type
  where
  CADropoutF _ 'WithDropout = Dropout
  CADropoutF _ 'WithoutDropout = ()

-- | Specifies the final layer normalization of the cross-attention layer.
type family
  CAFinalLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CAFinalLayerNormF 'T5 _ _ _ _ =
    ()
  CAFinalLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    CAFinalLayerNormF 'T5 gradient device dataType queryEmbedDim
  CAFinalLayerNormF 'BART gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  CAFinalLayerNormF 'MBART gradient device dataType queryEmbedDim =
    CAFinalLayerNormF 'BART gradient device dataType queryEmbedDim
  CAFinalLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    ()

-- | Specifies the parameters of a cross-attention layer.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @queryEmbedDim@: the dimension of the transformer query embeddings.
-- - @keyEmbedDim@: the dimension of the transformer key embeddings.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
crossAttentionSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim keyEmbedDim ->
  SHasDropout hasDropout ->
  Double ->
  Double ->
  ModelSpec (GCrossAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout)
crossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim hasDropout dropoutP eps =
  let initialLayerNormSpec ST5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SByT5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SBART = ()
      initialLayerNormSpec SMBART = ()
      initialLayerNormSpec SPegasus = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      initialLayerNormSpec SBERT = undefined
      initialLayerNormSpec SRoBERTa = undefined
      initialLayerNormSpec SGPT2 = undefined
      mhaSpec ST5 = NamedModel "EncDecAttention." $ mhaSpec' ST5
      mhaSpec SByT5 = NamedModel "EncDecAttention." $ mhaSpec' SByT5
      mhaSpec SBART = NamedModel "encoder_attn." $ mhaSpec' SBART
      mhaSpec SMBART = NamedModel "encoder_attn." $ mhaSpec' SMBART
      mhaSpec SPegasus = NamedModel "encoder_attn." $ mhaSpec' SPegasus
      mhaSpec SBERT = undefined
      mhaSpec SRoBERTa = undefined
      mhaSpec SGPT2 = undefined
      dropoutSpec _ SWithDropout = Dropout dropoutP
      dropoutSpec _ SWithoutDropout = ()
      finalLayerNormSpec ST5 = ()
      finalLayerNormSpec SByT5 = ()
      finalLayerNormSpec SBART = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SMBART = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SPegasus = ()
      finalLayerNormSpec SBERT = undefined
      finalLayerNormSpec SRoBERTa = undefined
      finalLayerNormSpec SGPT2 = undefined
   in GCrossAttention (initialLayerNormSpec style) (mhaSpec style) (dropoutSpec style hasDropout) (finalLayerNormSpec style)
  where
    mhaSpec' ::
      STransformerStyle style ->
      ModelSpec (GMultiHeadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim hasDropout)
    mhaSpec' style' = multiHeadAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim hasDropout dropoutP
    layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
    layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps

instance
  ( HasInitialize initialLayerNorm generatorDevice initialLayerNorm' generatorDevice0,
    HasInitialize multiHeadAttention generatorDevice0 multiHeadAttention' generatorDevice1,
    HasInitialize dropout generatorDevice1 dropout' generatorDevice2,
    HasInitialize finalLayerNorm generatorDevice2 finalLayerNorm' generatorOutputDevice
  ) =>
  HasInitialize
    (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    generatorDevice
    (GCrossAttention initialLayerNorm' multiHeadAttention' dropout' finalLayerNorm')
    generatorOutputDevice

instance
  ( HasStateDict initialLayerNorm,
    HasStateDict multiHeadAttention,
    HasStateDict dropout,
    HasStateDict finalLayerNorm
  ) =>
  HasStateDict (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)

-- | 'HasForward' instance for 'GCrossAttention'.
--
-- @
--        ┌───────┐    ┌─────┐    ┌───────────────┐
--        │ query │    │ key │    │ attentionBias │
--        └───┬───┘    └──┬──┘    └───────┬───────┘
--            │           │               │
-- ┌──────────┤           │               │
-- │          │           │               │
-- │          ▼           │               │
-- │ (caInitialLayerNorm) │               │
-- │          │           │               │
-- │          │       ┌───┴───┐           │
-- │          │       │       │           │
-- │          ▼       ▼       ▼           │
-- │        caMultiheadAttention◄─────────┘
-- │                  │
-- │                  ▼
-- │              caDropout
-- │                  │
-- └──────►add◄───────┘
--          │
--          ▼
--  (caFinalLayerNorm)
--          │
--          ▼
--      ┌───────┐
--      │ query │
--      └───────┘
-- @
instance
  ( HasForward
      initialLayerNorm
      (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      multiHeadAttention
      ( tensor0,
        Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
        Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
        Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      dropout
      tensor1
      generatorDevice1
      (Tensor gradient2 layout2 device2 dataType2 shape2)
      generatorDevice2,
    HasForward
      finalLayerNorm
      (Tensor (queryGradient <|> gradient2) (queryLayout <+> layout2) (queryDevice <+> device2) (queryDataType <+> dataType2) (BroadcastShapesF queryShape shape2))
      generatorDevice2
      output
      generatorOutputDevice,
    Catch (BroadcastShapesF queryShape shape2)
  ) =>
  HasForward
    (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    ( Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
      Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GCrossAttention {..} (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward caInitialLayerNorm
        >>>= (\query' -> IxStateT $ forward caMultiHeadAttention (query', key, key, attentionBias))
        >>>= IxStateT . forward caDropout
        >>>= ilift . (query `add`)
        >>>= IxStateT . forward caFinalLayerNorm
