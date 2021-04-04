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

module Torch.GraduallyTyped.NN.Transformer.T5.ThreeB where

import Control.Monad.Reader (ReaderT (runReaderT))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Model, T5ModelWithLMHead, T5Output, T5RelPosEncBucketDim)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-3B number of layers.
-- 'num_layers = 24'
type T5ThreeBNumLayers = 24

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
  (device :: Device (DeviceType Nat)) =
  T5Model T5ThreeBNumLayers device T5ThreeBHeadDim T5ThreeBHeadEmbedDim T5ThreeBEmbedDim T5ThreeBInputEmbedDim T5ThreeBFFNDim T5ThreeBVocabDim

-- | T5-3B model with language modelling head.
type T5ThreeBWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5ThreeBNumLayers device T5ThreeBHeadDim T5ThreeBHeadEmbedDim T5ThreeBEmbedDim T5ThreeBInputEmbedDim T5ThreeBFFNDim T5ThreeBVocabDim
