{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Base where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.NN.Transformer.BART.Common (BARTModel, BARTModelWithLMHead)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | BART-Base number of layers.
-- 'encoder_layers = 6'
-- 'decoder_layers = 6'
type BARTBaseNumLayers = 6

-- | BART-Base number of attention heads.
-- 'encoder_attention_heads = 12'
-- 'decoder_attention_heads = 12'
type BARTBaseHeadDim = 'Dim ('Name "*") ('Size 12)

-- | BART-Base head embedding dimension.
-- 'd_kv = 64'
type BARTBaseHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BART-Base embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type BARTBaseEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BART-Base model dimension.
-- 'd_model = 768'
type BARTBaseInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BART-Base feed-forward network dimension.
-- 'encoder_ffn_dim = 3072'
-- 'decoder_ffn_dim = 3072'
type BARTBaseFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | BART-Base vocabulary dimension.
-- 'vocab_size = 50265'
type BARTBaseVocabDim = 'Dim ('Name "*") ('Size 50265)

-- | BART-Base model.
type BARTBase
  (device :: Device (DeviceType Nat)) =
  BARTModel BARTBaseNumLayers device BARTBaseHeadDim BARTBaseHeadEmbedDim BARTBaseEmbedDim BARTBaseInputEmbedDim BARTBaseFFNDim BARTBaseVocabDim

-- | BART-Base model with language modelling head.
type BARTBaseWithLMHead
  (device :: Device (DeviceType Nat)) =
  BARTModelWithLMHead BARTBaseNumLayers device BARTBaseHeadDim BARTBaseHeadEmbedDim BARTBaseEmbedDim BARTBaseInputEmbedDim BARTBaseFFNDim BARTBaseVocabDim
