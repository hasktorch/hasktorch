{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeApplications #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus.XSum where

import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Pegasus.Common (PegasusModelF, pegasusModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | Pegasus-XSum number of layers.
-- 'encoder_layers = 16'
-- 'decoder_layers = 16'
type PegasusXSumNumLayers = 16

pegasusXSumNumLayers :: SNat PegasusXSumNumLayers
pegasusXSumNumLayers = sing @PegasusXSumNumLayers

-- | Pegasus-XSum number of attention heads.
-- 'encoder_attention_heads = 16'
-- 'decoder_attention_heads = 16'
type PegasusXSumHeadDim = 'Dim ('Name "*") ('Size 16)

-- | Pegasus-XSum head embedding dimension.
-- 'd_kv = 64'
type PegasusXSumHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | Pegasus-XSum embedding dimension.
-- 'hidden_size = n_heads * d_kv = 1024'
type PegasusXSumEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | Pegasus-XSum model dimension.
-- 'd_model = 1024'
type PegasusXSumInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

pegasusXSumInputEmbedDim :: SDim PegasusXSumInputEmbedDim
pegasusXSumInputEmbedDim = sing

-- | Pegasus-XSum feed-forward network dimension.
-- 'encoder_ffn_dim = 4096'
-- 'decoder_ffn_dim = 4096'
type PegasusXSumFFNDim = 'Dim ('Name "*") ('Size 4096)

-- | Pegasus-XSum vocabulary dimension.
-- 'vocab_size = 96103'
type PegasusXSumVocabDim = 'Dim ('Name "*") ('Size 96103)

-- | Pegasus-XSum vocabulary dimension singleton.
pegasusXSumVocabDim :: SDim PegasusXSumVocabDim
pegasusXSumVocabDim = sing

-- | Pegasus-XSum model.
type PegasusXSum
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  PegasusModelF transformerHead PegasusXSumNumLayers gradient device PegasusXSumHeadDim PegasusXSumHeadEmbedDim PegasusXSumEmbedDim PegasusXSumInputEmbedDim PegasusXSumFFNDim PegasusXSumVocabDim hasDropout

-- | Pegasus-XSum model specification.
pegasusXSumSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (PegasusXSum transformerHead gradient device hasDropout)
pegasusXSumSpec transformerHead = pegasusModelSpec transformerHead pegasusXSumNumLayers
