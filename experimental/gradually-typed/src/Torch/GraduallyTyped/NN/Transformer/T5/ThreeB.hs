{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.ThreeB where

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

-- | T5-3B number of layers.
-- 'num_layers = 24'
type T5ThreeBNumLayers = 24

-- | T5-3B number of layers singleton.
t5ThreeBNumLayers :: SNat T5ThreeBNumLayers
t5ThreeBNumLayers = sing

-- | T5-3B number of attention heads.
-- 'n_heads = 32'
type T5ThreeBHeadDim = 'Dim ('Name "*") ('Size 32)

-- | T5-3B head embedding dimension.
-- 'd_kv = 128'
type T5ThreeBHeadEmbedDim = 'Dim ('Name "*") ('Size 128)

-- | T5-3B embedding dimension.
-- 'inner_dim = n_heads * d_kv = 4096'
type T5ThreeBEmbedDim = 'Dim ('Name "*") ('Size 4096)

-- | T5-3B model dimension.
-- 'd_model = 1024'
type T5ThreeBInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | T5-3B feed-forward network dimension.
-- 'd_ff = 16384'
type T5ThreeBFFNDim = 'Dim ('Name "*") ('Size 16384)

-- | T5-3B vocabulary dimension.
-- 'vocab_size = 32128'
type T5ThreeBVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-3B model.
type T5ThreeB
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'T5 transformerHead T5ThreeBNumLayers T5ThreeBNumLayers gradient device T5ThreeBHeadDim T5ThreeBHeadEmbedDim T5ThreeBEmbedDim T5ThreeBInputEmbedDim T5ThreeBFFNDim T5ThreeBVocabDim hasDropout

-- | T5-3B model specification.
t5ThreeBSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (T5ThreeB transformerHead gradient device hasDropout)
t5ThreeBSpec transformerHead = t5ModelSpec ST5 transformerHead t5ThreeBNumLayers t5ThreeBNumLayers
