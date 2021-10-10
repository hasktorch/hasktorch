{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Torch.GraduallyTyped.NN.Transformer.BERT.BaseUncased where

import Data.Singletons (SingI (sing))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.NN.Class (ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.BERT.Common (BERTModelF, bertModelSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead, TransformerHead)
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))

-- | BERT-Base-Uncased number of layers.
-- 'num_hidden_layers = 12'
type BERTBaseUncasedNumLayers = 12

-- | BERT-Base-Uncased number of layers singleton.
bertBaseUncasedNumLayers :: SNat BERTBaseUncasedNumLayers
bertBaseUncasedNumLayers = sing

-- | BERT-Base-Uncased number of attention heads.
-- 'num_attention_heads = 12'
type BERTBaseUncasedHeadDim = 'Dim ('Name "*") ('Size 12)

-- | BERT-Base-Uncased number of attention heads singleton.
bertBaseUncasedHeadDim :: SDim BERTBaseUncasedHeadDim
bertBaseUncasedHeadDim = sing

-- | BERT-Base-Uncased head embedding dimension.
-- 'd_kv = 64'
type BERTBaseUncasedHeadEmbedDim = 'Dim ('Name "*") ('Size 64)

-- | BERT-Base-Uncased head embedding dimension singleton.
bertBaseUncasedHeadEmbedDim :: SDim BERTBaseUncasedHeadEmbedDim
bertBaseUncasedHeadEmbedDim = sing

-- | BERT-Base-Uncased embedding dimension.
-- 'hidden_size = n_heads * d_kv = 768'
type BERTBaseUncasedEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased embedding dimension singleton.
bertBaseUncasedEmbedDim :: SDim BERTBaseUncasedEmbedDim
bertBaseUncasedEmbedDim = sing

-- | BERT-Base-Uncased model dimension.
-- 'hidden_size = 768'
type BERTBaseUncasedInputEmbedDim = 'Dim ('Name "*") ('Size 768)

-- | BERT-Base-Uncased model dimension singleton.
bertBaseUncasedInputEmbedDim :: SDim BERTBaseUncasedInputEmbedDim
bertBaseUncasedInputEmbedDim = sing

-- | BERT-Base-Uncased feed-forward network dimension.
-- 'intermediate_size = 3072'
type BERTBaseUncasedFFNDim = 'Dim ('Name "*") ('Size 3072)

-- | BERT-Base-Uncased feed-forward network dimension singleton.
bertBaseUncasedFFNDim :: SDim BERTBaseUncasedFFNDim
bertBaseUncasedFFNDim = sing

-- | BERT-Base-Uncased vocabulary dimension.
-- 'vocab_size = 30522'
type BERTBaseUncasedVocabDim = 'Dim ('Name "*") ('Size 30522)

-- | BERT-Base-Uncased vocabulary dimension singleton.
bertBaseUncasedVocabDim :: SDim BERTBaseUncasedVocabDim
bertBaseUncasedVocabDim = sing

-- | BERT-Base-Uncased type vocabulary dimension.
-- 'type_vocab_size = 2'
type BERTBaseUncasedTypeVocabDim = 'Dim ('Name "*") ('Size 2)

-- | BERT-Base-Uncased type vocabulary dimension singleton.
bertBaseUncasedTypeVocabDim :: SDim BERTBaseUncasedTypeVocabDim
bertBaseUncasedTypeVocabDim = sing

-- | BERT-Base-Uncased model.
type BERTBaseUncased
  (transformerHead :: TransformerHead)
  (gradient :: Gradient RequiresGradient)
  (device :: Device (DeviceType Nat))
  (hasDropout :: HasDropout) =
  BERTModelF transformerHead BERTBaseUncasedNumLayers gradient device BERTBaseUncasedHeadDim BERTBaseUncasedHeadEmbedDim BERTBaseUncasedEmbedDim BERTBaseUncasedInputEmbedDim BERTBaseUncasedFFNDim BERTBaseUncasedVocabDim BERTBaseUncasedTypeVocabDim hasDropout

-- | BERT-Base-Uncased model specification.
bertBaseUnchasedSpec ::
  STransformerHead transformerHead ->
  SGradient gradient ->
  SDevice device ->
  SHasDropout hasDropout ->
  ModelSpec (BERTBaseUncased transformerHead gradient device hasDropout)
bertBaseUnchasedSpec transformerHead = bertModelSpec transformerHead bertBaseUncasedNumLayers
