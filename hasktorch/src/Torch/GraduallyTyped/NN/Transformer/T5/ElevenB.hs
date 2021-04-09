{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Model (..), T5ModelWithLMHead (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-11B number of layers.
-- 'num_layers = 24'
type T5ElevenBNumLayers = 24

-- | T5-11B number of attention heads.
-- 'n_heads = 128'
type T5ElevenBHeadDim = 'Dim ('Name "*") ('Size 128)

-- | T5-11B head embedding dimension.
-- 'd_kv = 128'
type T5ElevenBHeadEmbedDim = 'Dim ('Name "*") ('Size 128)

-- | T5-11B embedding dimension.
-- 'inner_dim = n_heads * d_kv = 16384'
type T5ElevenBEmbedDim = 'Dim ('Name "*") ('Size 16384)

-- | T5-11B model dimension.
-- 'd_model = 1024'
type T5ElevenBInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | T5-11B feed-forward network dimension.
-- 'd_ff = 65536'
type T5ElevenBFFNDim = 'Dim ('Name "*") ('Size 65536)

-- | T5-11B vocabulary dimension.
-- 'vocab_size = 32128'
type T5ElevenBVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-11B model.
type T5ElevenB
  (device :: Device (DeviceType Nat)) =
  T5Model T5ElevenBNumLayers device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5ElevenBVocabDim

-- | T5-11B model with language modelling head.
type T5ElevenBWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5ElevenBNumLayers device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5ElevenBVocabDim
