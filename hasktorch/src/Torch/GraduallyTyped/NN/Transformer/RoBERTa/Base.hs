{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common (RoBERTaModel)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | RoBERTa-Base number of layers.
-- 'num_hidden_layers = 12'
type RoBERTaBaseNumLayers = 12

-- | RoBERTa-Base number of attention heads.
-- 'num_attention_heads = 12'
type RoBERTaBaseHeadDim = 'Dim ('Name "*") ('Size 12)

-- | RoBERTa-Base head embedding dimension.
-- 'd_kv = 64'
type RoBERTaBaseHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | RoBERTa-Base embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type RoBERTaBaseEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | RoBERTa-Base model dimension.
-- 'hidden_size = 768'
type RoBERTaBaseInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | RoBERTa-Base feed-forward network dimension.
-- 'intermediate_size = 3072'
type RoBERTaBaseFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | RoBERTa-Base type vocabulary dimension.
-- 'type_vocab_size = 1'
type RoBERTaBaseTypeVocabDim = 'Dim ('Name "*") ('Size 1)

-- | RoBERTa-Base vocabulary dimension.
-- 'vocab_size = 50265'
type RoBERTaBaseVocabDim = 'Dim ('Name "*") ('Size 50265)

-- | RoBERTa-Base model.
type RoBERTaBase
  (device :: Device (DeviceType Nat)) =
  RoBERTaModel RoBERTaBaseNumLayers device RoBERTaBaseHeadDim RoBERTaBaseHeadEmbedDim RoBERTaBaseEmbedDim RoBERTaBaseInputEmbedDim RoBERTaBaseFFNDim RoBERTaBaseVocabDim RoBERTaBaseTypeVocabDim
