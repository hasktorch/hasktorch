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

module Torch.GraduallyTyped.NN.Transformer.T5.ElevenB where

import Control.Monad.Reader (ReaderT (runReaderT))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Model, T5ModelWithLMHead, T5Output, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformerWithLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-11B number of layers.
-- 'num_layers = 24'
type T5ElevenBNumLayers = 24

-- | T5-11B number of attention heads.
-- 'n_heads = 128'
type T5ElevenBHeadDim = 'Dim ('Name "*") ('Size 128)

-- | T5-11B head embedding dimension.
-- 'd_kv = 128'
type T5ElevenBHeadEmbedDim = 'Dim ('Name "*") ('Size 128)

-- | T5-11B embedding dimension.
-- 'inner_dim = n_heads * d_kv = 16384'
type T5ElevenBEmbedDim = 'Dim ('Name "*") ('Size 16384)

-- | T5-11B model dimension.
-- 'd_model = 1024'
type T5ElevenBInputEmbedDim = 'Dim ('Name "*") ('Size 1024)

-- | T5-11B feed-forward network dimension.
-- 'd_ff = 65536'
type T5ElevenBFFNDim = 'Dim ('Name "*") ('Size 65536)

-- | T5-11B vocabulary dimension.
-- 'vocab_size = 32128'
type T5ElevenBVocabDim = 'Dim ('Name "*") ('Size 32128)

-- | T5-11B configuration data type.
-- Modelled after https://huggingface.co/t5-11b/blob/main/config.json.
type T5ElevenBConfig device =
  T5Config T5ElevenBNumLayers device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5RelPosEncBucketDim T5ElevenBVocabDim

-- | load a T5-11B configuration from a file
t5ElevenBConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5ElevenBConfig ('Device 'CPU))
t5ElevenBConfigFromPretrained = t5ConfigFromPretrained

-- | T5-11B model.
type T5ElevenB
  (device :: Device (DeviceType Nat)) =
  T5Model T5ElevenBNumLayers device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5ElevenBVocabDim

-- | T5-11B model with language modelling head.
type T5ElevenBWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5ElevenBNumLayers device T5ElevenBHeadDim T5ElevenBHeadEmbedDim T5ElevenBEmbedDim T5ElevenBInputEmbedDim T5ElevenBFFNDim T5ElevenBVocabDim
