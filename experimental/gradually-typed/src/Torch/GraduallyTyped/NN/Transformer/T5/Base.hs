{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Base where

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

-- | T5-Base number of layers.
-- 'num_layers = 12'
type T5BaseNumLayers = 12

-- | T5-Base number of layers singleton.
t5BaseNumLayers :: SNat T5BaseNumLayers
t5BaseNumLayers = sing

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
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  T5ModelF 'T5 transformerHead T5BaseNumLayers T5BaseNumLayers gradient device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseVocabDim hasDropout

-- | T5-Base model specification.
t5BaseSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (T5Base transformerHead gradient device hasDropout)
t5BaseSpec transformerHead = t5ModelSpec ST5 transformerHead t5BaseNumLayers t5BaseNumLayers
