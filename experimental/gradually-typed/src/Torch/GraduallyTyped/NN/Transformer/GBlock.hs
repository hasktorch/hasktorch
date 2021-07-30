{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Transformer.GBlock where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Transformer.GCrossAttention (CADropoutF, CAFinalLayerNormF, CAInitialLayerNormF, CAMultiheadAttentionF, GCrossAttention, crossAttentionSpec)
import Torch.GraduallyTyped.NN.Transformer.GFeedForwardNetwork (FFNActivationDropoutF, FFNActivationF, FFNInputLayerNormF, FFNInputTransformationF, FFNOutputDropoutF, FFNOutputLayerNormF, FFNOutputProjectionF, GTransformerFeedForwardNetwork, transformerFeedForwardNetworkSpec)
import Torch.GraduallyTyped.NN.Transformer.GSelfAttention (GSelfAttention, SADropoutF, SAFinalLayerNormF, SAInitialLayerNormF, SAMultiheadAttentionF, selfAttentionSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | Generic transformer encoder block consisting of self-attention, cross-attention, and a feed-forward network.
--
-- - @selfAttention@ is a self-attention layer.
-- - @crossAttention@ is a cross-attention layer.
-- - @feedForwardNetwork@ is a feed-forward layer.
--
-- TODO: Some transformers use LayerDrop, see https://arxiv.org/abs/1909.11556, during training.
-- To support this, we will need a layer wrapper that is either the identity function or the wrapped layer
-- based on a uniformly random draw from a supplied generator.
data
  GTransformerBlock
    (selfAttention :: Type)
    (crossAttention :: Type)
    (feedForwardNetwork :: Type)
  where
  GTransformerBlock ::
    forall selfAttention crossAttention feedForwardNetwork.
    { -- | self-attention layer
      tbSelfAttention :: selfAttention,
      -- | cross-attention layer
      tbCrossAttention :: crossAttention,
      -- | feed-forward network
      tbFeedForwardNetwork :: feedForwardNetwork
    } ->
    GTransformerBlock selfAttention crossAttention feedForwardNetwork

type instance
  ModelSpec (GTransformerBlock selfAttention crossAttention feedForwardNetwork) =
    GTransformerBlock (ModelSpec selfAttention) (ModelSpec crossAttention) (ModelSpec feedForwardNetwork)

type family
  EncoderBlockSelfAttentionF
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
  EncoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim =
    NamedModel
      ( GSelfAttention
          (SAInitialLayerNormF style gradient device dataType queryEmbedDim)
          (SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
          (SADropoutF style)
          (SAFinalLayerNormF style gradient device dataType queryEmbedDim)
      )

type family EncoderBlockCrossAttentionF :: Type where
  EncoderBlockCrossAttentionF = ()

type family
  EncoderBlockFeedForwardNetworkF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EncoderBlockFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim =
    NamedModel
      ( GTransformerFeedForwardNetwork
          (FFNInputLayerNormF style gradient device dataType queryEmbedDim)
          (FFNInputTransformationF style gradient device dataType queryEmbedDim ffnDim)
          (FFNActivationF style)
          (FFNActivationDropoutF style)
          (FFNOutputProjectionF style gradient device dataType queryEmbedDim ffnDim)
          (FFNOutputDropoutF style)
          (FFNOutputLayerNormF style gradient device dataType queryEmbedDim)
      )

encoderBlockSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim ffnDim ->
  Double ->
  Double ->
  ModelSpec
    ( GTransformerBlock
        (EncoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
        EncoderBlockCrossAttentionF
        (EncoderBlockFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim)
    )
encoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps =
  let saSpec ST5 = NamedModel "layer.0." $ saSpec' ST5
      saSpec SByT5 = NamedModel "layer.0." $ saSpec' SByT5
      saSpec SBART = NamedModel mempty $ saSpec' SBART
      saSpec SMBART = NamedModel mempty $ saSpec' SMBART
      saSpec SPegasus = NamedModel mempty $ saSpec' SPegasus
      saSpec SBERT = NamedModel "attention." $ saSpec' SBERT
      saSpec SRoBERTa = NamedModel "attention." $ saSpec' SRoBERTa
      saSpec SGPT2 = undefined
      caSpec _ = ()
      ffnSpec ST5 = NamedModel "layer.1." $ ffnSpec' ST5
      ffnSpec SByT5 = NamedModel "layer.1." $ ffnSpec' SByT5
      ffnSpec SBART = NamedModel mempty $ ffnSpec' SBART
      ffnSpec SMBART = NamedModel mempty $ ffnSpec' SMBART
      ffnSpec SPegasus = NamedModel mempty $ ffnSpec' SPegasus
      ffnSpec SBERT = NamedModel mempty $ ffnSpec' SBERT
      ffnSpec SRoBERTa = NamedModel mempty $ ffnSpec' SRoBERTa
      ffnSpec SGPT2 = undefined
   in GTransformerBlock (saSpec style) (caSpec style) (ffnSpec style)
  where
    saSpec' :: _
    saSpec' style' = selfAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
    ffnSpec' :: _
    ffnSpec' style' = transformerFeedForwardNetworkSpec style' gradient device dataType queryEmbedDim ffnDim dropoutP eps

type family
  DecoderBlockSelfAttentionF
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
  DecoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim =
    NamedModel
      ( GSelfAttention
          (SAInitialLayerNormF style gradient device dataType queryEmbedDim)
          (SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
          (SADropoutF style)
          (SAFinalLayerNormF style gradient device dataType queryEmbedDim)
      )

type family
  DecoderBlockCrossAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  DecoderBlockCrossAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim =
    NamedModel
      ( GCrossAttention
          (CAInitialLayerNormF style gradient device dataType queryEmbedDim)
          (CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
          (CADropoutF style)
          (CAFinalLayerNormF style gradient device dataType queryEmbedDim)
      )

type family
  DecoderBlockFeedForwardNetworkF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  DecoderBlockFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim =
    NamedModel
      ( GTransformerFeedForwardNetwork
          (FFNInputLayerNormF style gradient device dataType queryEmbedDim)
          (FFNInputTransformationF style gradient device dataType queryEmbedDim ffnDim)
          (FFNActivationF style)
          (FFNActivationDropoutF style)
          (FFNOutputProjectionF style gradient device dataType queryEmbedDim ffnDim)
          (FFNOutputDropoutF style)
          (FFNOutputLayerNormF style gradient device dataType queryEmbedDim)
      )

decoderBlockSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim keyEmbedDim ->
  SDim ffnDim ->
  Double ->
  Double ->
  ModelSpec
    ( GTransformerBlock
        (DecoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
        (DecoderBlockCrossAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
        (DecoderBlockFeedForwardNetworkF style gradient device dataType queryEmbedDim ffnDim)
    )
decoderBlockSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps =
  let saSpec ST5 = NamedModel "layer.0." $ saSpec' ST5
      saSpec SByT5 = NamedModel "layer.0." $ saSpec' SByT5
      saSpec SBART = NamedModel mempty $ saSpec' SBART
      saSpec SMBART = NamedModel mempty $ saSpec' SMBART
      saSpec SPegasus = NamedModel mempty $ saSpec' SPegasus
      saSpec SBERT = undefined
      saSpec SRoBERTa = undefined
      saSpec SGPT2 = undefined
      caSpec ST5 = NamedModel "layer.1." $ caSpec' ST5
      caSpec SByT5 = NamedModel "layer.1." $ caSpec' SByT5
      caSpec SBART = NamedModel mempty $ caSpec' SBART
      caSpec SMBART = NamedModel mempty $ caSpec' SMBART
      caSpec SPegasus = NamedModel mempty $ caSpec' SPegasus
      caSpec SBERT = undefined
      caSpec SRoBERTa = undefined
      caSpec SGPT2 = undefined
      ffnSpec ST5 = NamedModel "layer.2." $ ffnSpec' ST5
      ffnSpec SByT5 = NamedModel "layer.2." $ ffnSpec' SByT5
      ffnSpec SBART = NamedModel mempty $ ffnSpec' SBART
      ffnSpec SMBART = NamedModel mempty $ ffnSpec' SMBART
      ffnSpec SPegasus = NamedModel mempty $ ffnSpec' SPegasus
      ffnSpec SBERT = undefined
      ffnSpec SRoBERTa = undefined
      ffnSpec SGPT2 = undefined
   in GTransformerBlock (saSpec style) (caSpec style) (ffnSpec style)
  where
    saSpec' :: _
    saSpec' style' = selfAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps
    caSpec' :: _
    caSpec' style' = crossAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps
    ffnSpec' :: _
    ffnSpec' style' = transformerFeedForwardNetworkSpec style' gradient device dataType queryEmbedDim ffnDim dropoutP eps

instance
  ( HasInitialize selfAttention generatorDevice selfAttention generatorDevice,
    HasInitialize crossAttention generatorDevice crossAttention generatorDevice,
    HasInitialize feedForwardNetwork generatorDevice feedForwardNetwork generatorDevice
  ) =>
  HasInitialize
    (GTransformerBlock selfAttention crossAttention feedForwardNetwork)
    generatorDevice
    (GTransformerBlock selfAttention crossAttention feedForwardNetwork)
    generatorDevice
  where
  initialize (GTransformerBlock saSpec caSpec ffnSpec) =
    let selfAttention = IxStateT . initialize $ saSpec
        crossAttention = IxStateT . initialize $ caSpec
        feedForwardNetwork = IxStateT . initialize $ ffnSpec
     in runIxStateT $ GTransformerBlock <<$>> selfAttention <<*>> crossAttention <<*>> feedForwardNetwork

instance
  ( HasStateDict selfAttention,
    HasStateDict crossAttention,
    HasStateDict feedForwardNetwork
  ) =>
  HasStateDict (GTransformerBlock selfAttention crossAttention feedForwardNetwork)
  where
  fromStateDict (GTransformerBlock saSpec caSpec ffnSpec) k =
    GTransformerBlock
      <$> fromStateDict saSpec k
      <*> fromStateDict caSpec k
      <*> fromStateDict ffnSpec k
  toStateDict k GTransformerBlock {..} = do
    () <- toStateDict k tbSelfAttention
    () <- toStateDict k tbCrossAttention
    () <- toStateDict k tbFeedForwardNetwork
    pure ()

-- | 'HasForward' instance for 'GTransformerBlock' in an encoder configuration.
--
-- @
--      ┌───────┐  ┌───────────────┐
--      │ query │  │ attentionBias │
--      └───┬───┘  └───────┬───────┘
--          │              │
--          ▼              │
--   tbSelfAttention◄──────┘
--          ▼
-- tbFeedForwardNetwork
--          │
--          ▼
--      ┌───────┐
--      │ query │
--      └───────┘
-- @
instance
  ( HasForward
      selfAttention
      (query, attentionBias)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      feedForwardNetwork
      tensor0
      generatorDevice0
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformerBlock selfAttention () feedForwardNetwork)
    (query, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformerBlock {..} (query, attentionBias) =
    runIxStateT $
      ireturn (query, attentionBias)
        >>>= IxStateT . forward tbSelfAttention
        >>>= IxStateT . forward tbFeedForwardNetwork

-- | 'HasForward' instance for 'GTransformerBlock' in a decoder configuration.
--
-- @
-- ┌──────────────────────┐  ┌───────┐  ┌─────┐  ┌────────────────────┐
-- │ decoderAttentionBias │  │ query │  │ key │  │ crossAttentionBias │
-- └──────────┬───────────┘  └───┬───┘  └──┬──┘  └─────────┬──────────┘
--            │                  │         │               │
--            │                  ▼         │               │
--            └──────────►tdbSelfAttention │               │
--                               │         │               │
--                               ▼         ▼               │
--                            tdbCrossAttention◄───────────┘
--                               │
--                               ▼
--                     tdbFeedForwardNetwork
--                               │
--                               ▼
--                           ┌───────┐
--                           │ query │
--                           └───────┘
-- @
instance
  ( HasForward
      selfAttention
      (query, attentionBias)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      crossAttention
      (tensor0, key, crossAttentionBias)
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      feedForwardNetwork
      tensor1
      generatorDevice1
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformerBlock selfAttention crossAttention feedForwardNetwork)
    (query, key, attentionBias, crossAttentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformerBlock {..} (query, key, attentionBias, crossAttentionBias) =
    runIxStateT $
      ireturn (query, attentionBias)
        >>>= IxStateT . forward tbSelfAttention
        >>>= (\query' -> IxStateT . forward tbCrossAttention $ (query', key, crossAttentionBias))
        >>>= IxStateT . forward tbFeedForwardNetwork
