{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Large where

import Data.Singletons (SingI (..))
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.BART.Common (BARTModelF, bartModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | BART-Large number of layers.
-- 'encoder_layers = 12'
-- 'decoder_layers = 12'
type BARTLargeNumLayers = 12

-- | BART-Large number of layers singleton.
bartLargeNumLayers :: SNat BARTLargeNumLayers
bartLargeNumLayers = sing

-- | BART-Large number of attention heads.
-- 'encoder_attention_heads = 16'
-- 'decoder_attention_heads = 16'
type BARTLargeHeadDim = 'Dim ('Name "*") ('Size 16)

-- | BART-Large number of attention heads singleton.
bartLargeHeadDim :: SDim BARTLargeHeadDim
bartLargeHeadDim = sing

-- | BART-Large head embedding dimension.
-- 'd_kv = 64'
type BARTLargeHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BART-Large head embedding dimension singleton.
bartLargeHeadEmbedDim :: SDim BARTLargeHeadEmbedDim
bartLargeHeadEmbedDim = sing

-- | BART-Large embedding dimension.
-- 'hidden_size = n_heads * d_kv = 1024'
type BARTLargeEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | BART-Large embedding dimension singleton.
bartLargeEmbedDim :: SDim BARTLargeEmbedDim
bartLargeEmbedDim = sing

-- | BART-Large model dimension.
-- 'd_model = 1024'
type BARTLargeInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | BART-Large model dimension singleton.
bartLargeInputEmbedDim :: SDim BARTLargeInputEmbedDim
bartLargeInputEmbedDim = sing

-- | BART-Large feed-forward network dimension.
-- 'encoder_ffn_dim = 4096'
-- 'decoder_ffn_dim = 4096'
type BARTLargeFFNDim = 'Dim ('Name "*") ('Size 4096)

-- | BART-Large feed-forward network dimension singleton.
bartLargeFFNDim :: SDim BARTLargeFFNDim
bartLargeFFNDim = sing

-- | BART-Large vocabulary dimension.
-- 'vocab_size = 50265'
type BARTLargeVocabDim = 'Dim ('Name "*") ('Size 50265)

-- | BART-Large vocabulary dimension singleton.
bartLargeVocabDim :: SDim BARTLargeVocabDim
bartLargeVocabDim = sing

-- | BART-Large model.
type BARTLarge
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  BARTModelF transformerHead BARTLargeNumLayers gradient device BARTLargeHeadDim BARTLargeHeadEmbedDim BARTLargeEmbedDim BARTLargeInputEmbedDim BARTLargeFFNDim BARTLargeVocabDim hasDropout

-- | BART-Large model specification.
bartLargeSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (BARTLarge transformerHead gradient device hasDropout)
bartLargeSpec transformerHead = bartModelSpec transformerHead bartLargeNumLayers
