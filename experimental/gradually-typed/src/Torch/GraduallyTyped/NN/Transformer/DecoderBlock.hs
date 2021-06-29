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
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderBlock where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention, lookupCrossAttention)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork, lookupTransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention, lookupSelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), KnownDim, Name (..), SDim, Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Transformer decoder block consisting of self-attention,
-- cross-attention, and a feed-forward network.
data
  TransformerDecoderBlock
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoderBlock ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tdbSelfAttention :: SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | cross-attention layer
      tdbCrossAttention :: CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
      -- | feed-forward network
      tdbFeedForwardNetwork :: TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

instance
  ( HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP),
    HasInitialize (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP),
    HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
  ) =>
  HasInitialize (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim queryEmbedDim ->
      SDim keyEmbedDim ->
      SDim ffnDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps = runState $ do
    selfAttention <-
      state $
        initialize
          @(SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
          device
          dataType
          headDim
          headEmbedDim
          embedDim
          queryEmbedDim
          dropoutP
          eps
    crossAttention <-
      state $
        initialize
          @(CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
          device
          dataType
          headDim
          headEmbedDim
          embedDim
          queryEmbedDim
          keyEmbedDim
          dropoutP
          eps
    feedForwardNetwork <-
      state $
        initialize
          @(TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
          device
          dataType
          queryEmbedDim
          ffnDim
          dropoutP
          eps
    pure $ TransformerDecoderBlock selfAttention crossAttention feedForwardNetwork

lookupDecoderBlock ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim keyEmbedDim,
    KnownDim ffnDim,
    Scalar dropoutP
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  Double ->
  String ->
  m (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
lookupDecoderBlock headDim headEmbedDim embedDim dropoutP eps prefix =
  let selfAttention ST5 = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "layer.0.")
      selfAttention SByT5 = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "layer.0.")
      selfAttention SBART = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix
      selfAttention SMBART = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix
      selfAttention SPegasus = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix
      selfAttention SBERT = undefined
      selfAttention SRoBERTa = undefined
      selfAttention SGPT2 = undefined
      crossAttention ST5 = lookupCrossAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "layer.1.")
      crossAttention SByT5 = lookupCrossAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "layer.1.")
      crossAttention SBART = lookupCrossAttention headDim headEmbedDim embedDim dropoutP eps prefix
      crossAttention SMBART = lookupCrossAttention headDim headEmbedDim embedDim dropoutP eps prefix
      crossAttention SPegasus = lookupCrossAttention headDim headEmbedDim embedDim dropoutP eps prefix
      crossAttention SBERT = undefined
      crossAttention SRoBERTa = undefined
      crossAttention SGPT2 = undefined
      feedForwardNetwork ST5 = lookupTransformerFeedForwardNetwork dropoutP eps (prefix <> "layer.2.")
      feedForwardNetwork SByT5 = lookupTransformerFeedForwardNetwork dropoutP eps (prefix <> "layer.2.")
      feedForwardNetwork SBART = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SMBART = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SPegasus = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SBERT = undefined
      feedForwardNetwork SRoBERTa = undefined
      feedForwardNetwork SGPT2 = undefined
   in TransformerDecoderBlock
        <$> selfAttention (sing @style)
        <*> crossAttention (sing @style)
        <*> feedForwardNetwork (sing @style)

-- | 'HasForward' instance for 'TransformerDecoderBlock'.
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
      (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor decoderAttentionBiasRequiresGradient decoderAttentionBiasLayout decoderAttentionBiasDevice decoderAttentionBiasDataType decoderAttentionBiasShape
      )
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
      ( selfAttentionOutput,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor crossAttentionBiasRequiresGradient crossAttentionBiasLayout crossAttentionBiasDevice crossAttentionBiasDataType crossAttentionBiasShape
      )
      selfAttentionGeneratorOutput
      crossAttentionOutput
      crossAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
      crossAttentionOutput
      crossAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoderBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor decoderAttentionBiasRequiresGradient decoderAttentionBiasLayout decoderAttentionBiasDevice decoderAttentionBiasDataType decoderAttentionBiasShape,
      Tensor crossAttentionBiasRequiresGradient crossAttentionBiasLayout crossAttentionBiasDevice crossAttentionBiasDataType crossAttentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerDecoderBlock {..} (query, key, decoderAttentionBias, crossAttentionBias) =
    runIxState $
      ireturn (query, decoderAttentionBias)
        >>>= IxState . forward tdbSelfAttention
        >>>= (\query' -> IxState . forward tdbCrossAttention $ (query', key, crossAttentionBias))
        >>>= IxState . forward tdbFeedForwardNetwork
