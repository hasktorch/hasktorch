{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Base where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Model (..), T5ModelWithLMHead (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Base number of layers.
-- 'num_layers = 12'
type T5BaseNumLayers = 12

-- | T5-Base number of attention heads.
-- 'n_heads = 12'
type T5BaseHeadDim = 'Dim ('Name "*") ('Size 12)

-- | T5-Base head embedding dimension.
-- 'd_kv = 64'
type T5BaseHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | T5-Base embedding dimension.
-- 'inner_dim = n_heads * d_kv = 768'
type T5BaseEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | T5-Base model dimension.
-- 'd_model = 768'
type T5BaseInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | T5-Base feed-forward network dimension.
-- 'd_ff = 3072'
type T5BaseFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | T5-Base vocabulary dimension.
-- 'vocab_size = 32128'
type T5BaseVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-Base model.
type T5Base
  (device :: Device (DeviceType Nat)) =
  T5Model T5BaseNumLayers T5BaseNumLayers device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseVocabDim

-- | T5-Base model with language modelling head.
type T5BaseWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5BaseNumLayers T5BaseNumLayers device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseVocabDim
