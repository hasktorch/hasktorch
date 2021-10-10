{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Base where

import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.BART.Common (BARTModelF, bartModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | BART-Base number of layers.
-- 'encoder_layers = 6'
-- 'decoder_layers = 6'
type BARTBaseNumLayers = 6

-- | BART-Base number of layers singleton.
bartBaseNumLayers :: SNat BARTBaseNumLayers
bartBaseNumLayers = sing

-- | BART-Base number of attention heads.
-- 'encoder_attention_heads = 12'
-- 'decoder_attention_heads = 12'
type BARTBaseHeadDim = 'Dim ('Name "*") ('Size 12)

-- | BART-Base number of attention heads singleton.
bartBaseHeadDim :: SDim BARTBaseHeadDim
bartBaseHeadDim = sing

-- | BART-Base head embedding dimension.
-- 'd_kv = 64'
type BARTBaseHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BART-Base head embedding dimension singleton.
bartBaseHeadEmbedDim :: SDim BARTBaseHeadEmbedDim
bartBaseHeadEmbedDim = sing

-- | BART-Base embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type BARTBaseEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BART-Base embedding dimension singleton.
bartBaseEmbedDim :: SDim BARTBaseEmbedDim
bartBaseEmbedDim = sing

-- | BART-Base model dimension.
-- 'd_model = 768'
type BARTBaseInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BART-Base model dimension singleton.
bartBaseInputEmbedDim :: SDim BARTBaseInputEmbedDim
bartBaseInputEmbedDim = sing

-- | BART-Base feed-forward network dimension.
-- 'encoder_ffn_dim = 3072'
-- 'decoder_ffn_dim = 3072'
type BARTBaseFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | BART-Base feed-forward network dimension singleton.
bartBaseFFNDim :: SDim BARTBaseFFNDim
bartBaseFFNDim = sing

-- | BART-Base vocabulary dimension.
-- 'vocab_size = 50265'
type BARTBaseVocabDim = 'Dim ('Name "*") ('Size 50265)

-- | BART-Base vocabulary dimension singleton.
bartBaseVocabDim :: SDim BARTBaseVocabDim
bartBaseVocabDim = sing

-- | BART-Base model.
type BARTBase
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  BARTModelF transformerHead BARTBaseNumLayers gradient device BARTBaseHeadDim BARTBaseHeadEmbedDim BARTBaseEmbedDim BARTBaseInputEmbedDim BARTBaseFFNDim BARTBaseVocabDim hasDropout

-- | BART-Base model specification.
bartBaseSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (BARTBase transformerHead gradient device hasDropout)
bartBaseSpec transformerHead = bartModelSpec transformerHead bartBaseNumLayers
