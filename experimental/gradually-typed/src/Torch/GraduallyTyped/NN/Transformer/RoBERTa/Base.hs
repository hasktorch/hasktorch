{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.RoBERTa.Base where

import Data.Singletons (SingI (sing))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.RoBERTa.Common (RoBERTaModelF, robertaModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | RoBERTa-Base number of layers.
-- 'num_hidden_layers = 12'
type RoBERTaBaseNumLayers = 12

-- | RoBERTa-Base number of layers singleton.
robertaBaseNumLayers :: SNat RoBERTaBaseNumLayers
robertaBaseNumLayers = sing

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

-- | RoBERTa-Base vocabulary dimension.
-- 'vocab_size = 50265'
type RoBERTaBaseVocabDim = 'Dim ('Name "*") ('Size 50265)

-- | RoBERTa-Base vocabulary dimension singleton.
robertaBaseVocabDim :: SDim RoBERTaBaseVocabDim
robertaBaseVocabDim = sing

-- | RoBERTa-Base type vocabulary dimension.
-- 'type_vocab_size = 1'
type RoBERTaBaseTypeVocabDim = 'Dim ('Name "*") ('Size 1)

-- | RoBERTa-Base model.
type RoBERTaBase
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  RoBERTaModelF transformerHead RoBERTaBaseNumLayers gradient device RoBERTaBaseHeadDim RoBERTaBaseHeadEmbedDim RoBERTaBaseEmbedDim RoBERTaBaseInputEmbedDim RoBERTaBaseFFNDim RoBERTaBaseVocabDim RoBERTaBaseTypeVocabDim hasDropout

-- | RoBERTa-Base model specification.
robertaBaseSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (RoBERTaBase transformerHead gradient device hasDropout)
robertaBaseSpec transformerHead = robertaModelSpec transformerHead robertaBaseNumLayers
