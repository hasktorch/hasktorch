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
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | Transformer encoder block consisting of self-attention and a feed-forward network.
--
-- TODO: Some transformers use LayerDrop, see https://arxiv.org/abs/1909.11556, during training.
-- To support this, we will need a layer wrapper that is either the identity function or the wrapped layer
-- based on a uniformly random draw from a supplied generator.
-- Complications will arise due to the gradual typing...
data
  TransformerBlock
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
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
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    { -- | self-attention layer
      tbSelfAttention :: SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
      -- | feed-forward network
      tbFeedForwardNetwork :: TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

instance
  ( HasInitialize
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, dropoutP, Double)
      generator
      generator',
    HasInitialize
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim, dropoutP, Double)
      generator'
      generator''
  ) =>
  HasInitialize
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim ffnDim, dropoutP, Double)
    generator
    generator''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, ffnDim, dropoutP, eps) =
    let selfAttention = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps)
        feedForwardNetwork = IxState . initialize $ (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps)
     in runIxState $ TransformerBlock <<$>> selfAttention <<*>> feedForwardNetwork

instance
  SingI style =>
  HasStateDict
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim ffnDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, ffnDim, dropoutP, eps) k =
    let selfAttention ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "layer.0.")
        selfAttention SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "layer.0.")
        selfAttention SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) k
        selfAttention SBERT = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "attention.")
        selfAttention SRoBERTa = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, dropoutP, eps) (k <> "attention.")
        selfAttention SGPT2 = undefined
        feedForwardNetwork ST5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) (k <> "layer.1.")
        feedForwardNetwork SByT5 = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) (k <> "layer.1.")
        feedForwardNetwork SBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SMBART = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SPegasus = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SBERT = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SRoBERTa = fromStateDict (gradient, device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) k
        feedForwardNetwork SGPT2 = undefined
     in TransformerBlock
          <$> selfAttention (sing @style)
          <*> feedForwardNetwork (sing @style)
  toStateDict k TransformerBlock {..} =
    let selfAttention ST5 = toStateDict (k <> "layer.0.")
        selfAttention SByT5 = toStateDict (k <> "layer.0.")
        selfAttention SBART = toStateDict k
        selfAttention SMBART = toStateDict k
        selfAttention SPegasus = toStateDict k
        selfAttention SBERT = toStateDict (k <> "attention.")
        selfAttention SRoBERTa = toStateDict (k <> "attention.")
        selfAttention SGPT2 = undefined
        feedForwardNetwork ST5 = toStateDict (k <> "layer.1.")
        feedForwardNetwork SByT5 = toStateDict (k <> "layer.1.")
        feedForwardNetwork SBART = toStateDict k
        feedForwardNetwork SMBART = toStateDict k
        feedForwardNetwork SPegasus = toStateDict k
        feedForwardNetwork SBERT = toStateDict k
        feedForwardNetwork SRoBERTa = toStateDict k
        feedForwardNetwork SGPT2 = undefined
     in do
          () <- selfAttention (sing @style) tbSelfAttention
          () <- feedForwardNetwork (sing @style) tbFeedForwardNetwork
          pure ()

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
      (SelfAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
      input
      (Generator generatorDevice)
      selfAttentionOutput
      selfAttentionGeneratorOutput,
    HasForward
      (TransformerFeedForwardNetwork style gradient device dataType queryEmbedDim ffnDim dropoutP)
      selfAttentionOutput
      selfAttentionGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerBlock style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    input
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward TransformerBlock {..} input =
    runIxState $
      ireturn input
        >>>= IxState . forward tbSelfAttention
        >>>= IxState . forward tbFeedForwardNetwork
