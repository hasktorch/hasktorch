{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Large where

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

-- | T5-Large number of layers.
-- 'num_layers = 24'
type T5LargeNumLayers = 24

-- | T5-Large number of layers singleton.
t5LargeNumLayers :: SNat T5LargeNumLayers
t5LargeNumLayers = sing

-- | T5-Large number of attention heads.
-- 'n_heads = 16'
type T5LargeHeadDim = 'Dim ('Name "*") ('Size 16)

-- | T5-Large head embedding dimension.
-- 'd_kv = 64'
type T5LargeHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | T5-Large embedding dimension.
-- 'inner_dim = n_heads * d_kv = 1024'
type T5LargeEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | T5-Large model dimension.
-- 'd_model = 1024'
type T5LargeInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | T5-Large feed-forward network dimension.
-- 'd_ff = 4096'
type T5LargeFFNDim = 'Dim ('Name "*") ('Size 4096)

-- | T5-Large vocabulary dimension.
-- 'vocab_size = 32128'
type T5LargeVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-Large model.
type T5Large
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'T5 transformerHead T5LargeNumLayers T5LargeNumLayers gradient device T5LargeHeadDim T5LargeHeadEmbedDim T5LargeEmbedDim T5LargeInputEmbedDim T5LargeFFNDim T5LargeVocabDim hasDropout

-- | T5-Large model specification.
t5LargeSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (T5Large transformerHead gradient device hasDropout)
t5LargeSpec transformerHead = t5ModelSpec ST5 transformerHead t5LargeNumLayers t5LargeNumLayers
