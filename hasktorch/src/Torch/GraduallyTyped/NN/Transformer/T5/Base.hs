{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Base where

import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Coerce (coerce)
import Data.Kind (Type)
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Model (..), T5ModelWithLMHead (..), T5Output, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformer, lookupSequenceToSequenceTransformerWithLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (T5))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Base number of layers.
-- 'num_layers = 12'
type T5BaseNumLayers = 12

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

-- | T5-Base configuration data type.
-- Modelled after https://huggingface.co/t5-base/blob/main/config.json.
type T5BaseConfig device =
  T5Config T5BaseNumLayers device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5RelPosEncBucketDim T5BaseVocabDim

-- | load a T5-Base configuration from a file
t5BaseConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5BaseConfig ('Device 'CPU))
t5BaseConfigFromPretrained = t5ConfigFromPretrained

-- | T5-Base model.
type T5Base
  (device :: Device (DeviceType Nat)) =
  T5Model T5BaseNumLayers device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseVocabDim

-- | T5-Base model with language modelling head.
type T5BaseWithLMHead
  (device :: Device (DeviceType Nat)) =
  T5ModelWithLMHead T5BaseNumLayers device T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseVocabDim
