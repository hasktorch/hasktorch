{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# OPTIONS_GHC -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Small where

import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Model (..), T5ModelWithLMHead (..), ByT5Model, ByT5ModelWithLMHead)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Small number of layers.
-- 'num_layers = 6'
type T5SmallNumLayers = 6

-- | T5-Small number of attention heads.
-- 'n_heads = 8'
type T5SmallHeadDim = 'Dim ('Name "*") ('Size 8)

-- | T5-Small head embedding dimension.
-- 'd_kv = 64'
type T5SmallHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | T5-Small embedding dimension.
-- 'inner_dim = n_heads * d_kv = 512'
type T5SmallEmbedDim = 'Dim ('Name "*") ('Size 512)

-- | T5-Small model dimension.
-- 'd_model = 512'
type T5SmallInputEmbedDim = 'Dim ('Name "*") ('Size 512)

-- | T5-Small feed-forward network dimension.
-- 'd_ff = 2048'
type T5SmallFFNDim = 'Dim ('Name "*") ('Size 2048)

-- | T5-Small vocabulary dimension.
-- 'vocab_size = 32128'
type T5SmallVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-Small model.
type T5Small
  (device :: Device (DeviceType Nat)) =
  T5Model T5SmallNumLayers T5SmallNumLayers device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallVocabDim

-- | T5-Small model with language modelling head.
type T5SmallWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5SmallNumLayers T5SmallNumLayers device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallVocabDim

-- | ByT5-Small number of encoder layers.
-- 'num_layers = 12'
type ByT5SmallNumEncoderLayers = 12

-- | ByT5-Small number of decoder layers.
-- 'num_decoder_layers = 4'
type ByT5SmallNumDecoderLayers = 4

-- | ByT5-Small number of attention heads.
-- 'n_heads = 6'
type ByT5SmallHeadDim = 'Dim ('Name "*") ('Size 6)

-- | ByT5-Small head embedding dimension.
-- 'd_kv = 64'
type ByT5SmallHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | ByT5-Small embedding dimension.
-- 'inner_dim = n_heads * d_kv = 384'
type ByT5SmallEmbedDim = 'Dim ('Name "*") ('Size 384)

-- | ByT5-Small model dimension.
-- 'd_model = 1472'
type ByT5SmallInputEmbedDim = 'Dim ('Name "*") ('Size 1472)

-- | T5-Small feed-forward network dimension.
-- 'd_ff = 3584'
type ByT5SmallFFNDim = 'Dim ('Name "*") ('Size 3584)

-- | T5-Small vocabulary dimension.
-- 'vocab_size = 384'
type ByT5SmallVocabDim = 'Dim ('Name "*") ('Size 384)

-- | ByT5-Small model.
type ByT5Small
  (device :: Device (DeviceType Nat)) =
  ByT5Model ByT5SmallNumEncoderLayers ByT5SmallNumDecoderLayers device ByT5SmallHeadDim ByT5SmallHeadEmbedDim ByT5SmallEmbedDim ByT5SmallInputEmbedDim ByT5SmallFFNDim ByT5SmallVocabDim

-- | ByT5-Small model with language modelling head.
type ByT5SmallWithLMHead
  (device :: Device (DeviceType Nat)) =
  ByT5ModelWithLMHead ByT5SmallNumEncoderLayers ByT5SmallNumDecoderLayers device ByT5SmallHeadDim ByT5SmallHeadEmbedDim ByT5SmallEmbedDim ByT5SmallInputEmbedDim ByT5SmallFFNDim ByT5SmallVocabDim