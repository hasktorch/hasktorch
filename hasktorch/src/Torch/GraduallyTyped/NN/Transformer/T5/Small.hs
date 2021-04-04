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

module Torch.GraduallyTyped.NN.Transformer.T5.Small where

import Control.Monad.Reader (ReaderT (runReaderT))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerInput)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Model, T5ModelWithLMHead, T5Output, T5RelPosEncBucketDim)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Small number of layers.
-- 'num_layers = 6'
type T5SmallNumLayers = 6

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

-- | T5-Small model.
type T5Small
  (device :: Device (DeviceType Nat)) =
  T5Model T5SmallNumLayers device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallVocabDim

-- | T5-Small model with language modelling head.
type T5SmallWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5SmallNumLayers device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallVocabDim
