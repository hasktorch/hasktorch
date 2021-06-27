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

module Torch.GraduallyTyped.NN.Transformer.Block where

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
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork, lookupTransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention, lookupSelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, Size (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Transformer encoder block consisting of self-attention and a feed-forward network.
--
-- TODO: Some transformers use LayerDrop, see https://arxiv.org/abs/1909.11556, during training.
-- To support this, we will need a layer wrapper that is either the identity function or the wrapped layer
-- based on a uniformly random draw from a supplied generator.
-- Complications will arise due to the gradual typing...
data
  TransformerBlock
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerBlock ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tbSelfAttention :: SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | feed-forward network
      tbFeedForwardNetwork :: TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

instance
  ( HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP),
    HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
  ) =>
  HasInitialize (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim queryEmbedDim ->
      SDim ffnDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps =
    runState $ do
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
      pure $ TransformerBlock selfAttention feedForwardNetwork

lookupBlock ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim ffnDim,
    Scalar dropoutP
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  Double ->
  String ->
  m (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
lookupBlock headDim headEmbedDim embedDim dropoutP eps prefix =
  let selfAttention ST5 = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "layer.0.")
      selfAttention SBERT = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "attention.")
      selfAttention SRoBERTa = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps (prefix <> "attention.")
      selfAttention SPegasus = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix
      selfAttention SBART = lookupSelfAttention headDim headEmbedDim embedDim dropoutP eps prefix
      feedForwardNetwork ST5 = lookupTransformerFeedForwardNetwork dropoutP eps (prefix <> "layer.1.")
      feedForwardNetwork SBERT = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SRoBERTa = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SPegasus = lookupTransformerFeedForwardNetwork dropoutP eps prefix
      feedForwardNetwork SBART = lookupTransformerFeedForwardNetwork dropoutP eps prefix
   in TransformerBlock
        <$> selfAttention (sing @style)
        <*> feedForwardNetwork (sing @style)

-- | 'HasForward' instance for 'TransformerBlock'.
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
      (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
      selfAttentionOutput
      selfAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerBlock style device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerBlock {..} (query, attentionBias) =
    runIxState $
      ireturn (query, attentionBias)
        >>>= IxState . forward tbSelfAttention
        >>>= IxState . forward tbFeedForwardNetwork
