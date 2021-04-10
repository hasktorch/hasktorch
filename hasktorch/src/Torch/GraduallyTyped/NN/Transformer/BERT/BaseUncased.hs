{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.NN.Transformer.BERT.Common (BERTModel, BERTModelWithLMHead)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | BERT-Base-Uncased number of layers.
-- 'num_hidden_layers = 12'
type BERTBaseUncasedNumLayers = 12

-- | BERT-Base-Uncased number of attention heads.
-- 'num_attention_heads = 12'
type BERTBaseUncasedHeadDim = 'Dim ('Name "*") ('Size 12)

-- | BERT-Base-Uncased head embedding dimension.
-- 'd_kv = 64'
type BERTBaseUncasedHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BERT-Base-Uncased embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type BERTBaseUncasedEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased model dimension.
-- 'hidden_size = 768'
type BERTBaseUncasedInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased feed-forward network dimension.
-- 'intermediate_size = 3072'
type BERTBaseUncasedFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | BERT-Base-Uncased vocabulary dimension.
-- 'vocab_size = 30522'
type BERTBaseUncasedVocabDim = 'Dim ('Name "*") ('Size 30522)

-- | BERT-Base-Uncased type vocabulary dimension.
-- 'type_vocab_size = 2'
type BERTBaseUncasedTypeVocabDim = 'Dim ('Name "*") ('Size 2)

-- | BERT-Base-Uncased model.
type BERTBaseUncased
  (device :: Device (DeviceType Nat)) =
  BERTModel BERTBaseUncasedNumLayers device BERTBaseUncasedHeadDim BERTBaseUncasedHeadEmbedDim BERTBaseUncasedEmbedDim BERTBaseUncasedInputEmbedDim BERTBaseUncasedFFNDim BERTBaseUncasedVocabDim BERTBaseUncasedTypeVocabDim

-- | BERT-Base-Uncased model with language modelling head.
type BERTBaseUncasedWithLMHead
  (device :: Device (DeviceType Nat)) =
  BERTModelWithLMHead BERTBaseUncasedNumLayers device BERTBaseUncasedHeadDim BERTBaseUncasedHeadEmbedDim BERTBaseUncasedEmbedDim BERTBaseUncasedInputEmbedDim BERTBaseUncasedFFNDim BERTBaseUncasedVocabDim BERTBaseUncasedTypeVocabDim
