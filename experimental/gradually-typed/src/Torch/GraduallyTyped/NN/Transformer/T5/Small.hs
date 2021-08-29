{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Small where

import Data.Singletons (SingI (sing))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5ModelF, t5ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, STransformerStyle (SByT5, ST5), TransformerHead, TransformerStyle (ByT5, T5))
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | T5-Small number of layers.
-- 'num_layers = 6'
type T5SmallNumLayers = 6

-- | T5-Small number of layers singleton.
t5SmallNumLayers :: SNat T5SmallNumLayers
t5SmallNumLayers = sing

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

t5SmallVocabDim :: SDim T5SmallVocabDim
t5SmallVocabDim = sing

-- | T5-Small model.
type T5Small
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'T5 transformerHead T5SmallNumLayers T5SmallNumLayers gradient device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallVocabDim hasDropout

-- | T5-Small model specification.
t5SmallSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (T5Small transformerHead gradient device hasDropout)
t5SmallSpec transformerHead = t5ModelSpec ST5 transformerHead t5SmallNumLayers t5SmallNumLayers

-- | ByT5-Small number of encoder layers.
-- 'num_layers = 12'
type ByT5SmallNumEncoderLayers = 12

-- | ByT5-Small number of encoder layers singleton.
byT5SmallNumEncoderLayers :: SNat ByT5SmallNumEncoderLayers
byT5SmallNumEncoderLayers = sing

-- | ByT5-Small number of decoder layers.
-- 'num_decoder_layers = 4'
type ByT5SmallNumDecoderLayers = 4

-- | ByT5-Small number of encoder layers singleton.
byT5SmallNumDecoderLayers :: SNat ByT5SmallNumDecoderLayers
byT5SmallNumDecoderLayers = sing

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
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'ByT5 transformerHead ByT5SmallNumEncoderLayers ByT5SmallNumDecoderLayers gradient device ByT5SmallHeadDim ByT5SmallHeadEmbedDim ByT5SmallEmbedDim ByT5SmallInputEmbedDim ByT5SmallFFNDim ByT5SmallVocabDim hasDropout

-- | ByT5-Small model specification.
byT5SmallSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (ByT5Small transformerHead gradient device hasDropout)
byT5SmallSpec transformerHead = t5ModelSpec SByT5 transformerHead byT5SmallNumEncoderLayers byT5SmallNumDecoderLayers
