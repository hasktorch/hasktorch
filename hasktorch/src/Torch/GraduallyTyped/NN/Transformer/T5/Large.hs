{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Large where

import Control.Monad.Reader (ReaderT (runReaderT))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Model, T5ModelWithLMHead, T5Output, T5RelPosEncBucketDim)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Large number of layers.
-- 'num_layers = 24'
type T5LargeNumLayers = 24

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
  (device :: Device (DeviceType Nat)) =
  T5Model T5LargeNumLayers device T5LargeHeadDim T5LargeHeadEmbedDim T5LargeEmbedDim T5LargeInputEmbedDim T5LargeFFNDim T5LargeVocabDim

-- | T5-Large model with language modelling head.
type T5LargeWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5LargeNumLayers device T5LargeHeadDim T5LargeHeadEmbedDim T5LargeEmbedDim T5LargeInputEmbedDim T5LargeFFNDim T5LargeVocabDim
