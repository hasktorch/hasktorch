{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB where

import Data.Singletons (SingI (sing))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5ModelF, t5ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, STransformerStyle (ST5), TransformerHead, TransformerStyle (T5))
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-11B number of layers.
-- 'num_layers = 24'
type T5ElevenBNumLayers = 24

-- | T5-11B number of layers singleton.
t5ElevenBNumLayers :: SNat T5ElevenBNumLayers
t5ElevenBNumLayers = sing

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
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'T5 transformerHead T5ElevenBNumLayers T5ElevenBNumLayers gradient device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5ElevenBVocabDim hasDropout

-- | T5-11B model specification.
t5ElevenBSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (T5ElevenB transformerHead gradient device hasDropout)
t5ElevenBSpec transformerHead = t5ModelSpec ST5 transformerHead t5ElevenBNumLayers t5ElevenBNumLayers
