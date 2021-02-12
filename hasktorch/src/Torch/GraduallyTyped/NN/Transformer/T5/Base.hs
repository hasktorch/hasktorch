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

module Torch.GraduallyTyped.NN.Transformer.T5.Base where

import Control.Monad.Reader (ReaderT (runReaderT))
import GHC.TypeLits (Nat)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5GenerationInput, T5Input, T5Output, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformerWithoutLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Base number of layers.
-- 'num_layers = 12'
type T5BaseNumLayers = 12

-- | T5-Base number of attention heads.
-- 'n_heads = 12'
type T5BaseHeadDim = 'Dim ( 'Name "*") ( 'Size 12)

-- | T5-Base head embedding dimension.
-- 'd_kv = 64'
type T5BaseHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | T5-Base embedding dimension.
-- 'inner_dim = n_heads * d_kv = 768'
type T5BaseEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

-- | T5-Base model dimension.
-- 'd_model = 768'
type T5BaseInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

-- | T5-Base feed-forward network dimension.
-- 'd_ff = 3072'
type T5BaseFFNDim = 'Dim ( 'Name "*") ( 'Size 3072)

-- | T5-Base vocabulary dimension.
-- 'vocab_size = 32128'
type T5BaseVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

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
  IO (T5BaseConfig ( 'Device 'CPU))
t5BaseConfigFromPretrained = t5ConfigFromPretrained

-- | T5-Base data type.
data
  T5Base
    (hasLMHead :: HasLMHead)
    (device :: Device (DeviceType Nat))
  where
  -- | T5-Base constructor.
  T5Base ::
    forall hasLMHead device.
    { t5BaseSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5BaseNumLayers
          T5BaseNumLayers
          device
          T5DataType
          T5BaseHeadDim
          T5BaseHeadEmbedDim
          T5BaseEmbedDim
          T5BaseInputEmbedDim
          T5BaseFFNDim
          T5RelPosEncBucketDim
          T5BaseVocabDim
          T5DropoutP
    } ->
    T5Base hasLMHead device

instance HasInitialize (T5Base 'WithoutLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5Base 'WithoutLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5Base 'WithoutLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5BaseConfigFromPretrained filePath False
    flip runReaderT config $
      T5Base <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5Base 'WithLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5Base 'WithLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5Base 'WithLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5BaseConfigFromPretrained filePath False
    flip runReaderT config $
      T5Base <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5BaseNumLayers
        T5BaseNumLayers
        device
        T5DataType
        T5BaseHeadDim
        T5BaseHeadEmbedDim
        T5BaseEmbedDim
        T5BaseInputEmbedDim
        T5BaseFFNDim
        T5RelPosEncBucketDim
        T5BaseVocabDim
        T5DropoutP
    )
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput =>
  HasForward
    (T5Base hasLMHead device)
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput
  where
  forward T5Base {..} = forward t5BaseSeqToSeq

instance
  ( HasForward
      ( SequenceToSequenceTransformer
          hasLMHead
          T5BaseNumLayers
          T5BaseNumLayers
          device
          T5DataType
          T5BaseHeadDim
          T5BaseHeadEmbedDim
          T5BaseEmbedDim
          T5BaseInputEmbedDim
          T5BaseFFNDim
          T5RelPosEncBucketDim
          T5BaseVocabDim
          T5DropoutP
      )
      (T5Input input decoderInput)
      generator
      output
      generatorOutput
  ) =>
  HasForward
    (T5Base hasLMHead device)
    (T5Input input decoderInput)
    generator
    output
    generatorOutput
  where
  forward T5Base {..} = forward t5BaseSeqToSeq

instance
  ( HasForward
      ( SequenceToSequenceTransformer
          hasLMHead
          T5BaseNumLayers
          T5BaseNumLayers
          device
          T5DataType
          T5BaseHeadDim
          T5BaseHeadEmbedDim
          T5BaseEmbedDim
          T5BaseInputEmbedDim
          T5BaseFFNDim
          T5RelPosEncBucketDim
          T5BaseVocabDim
          T5DropoutP
      )
      (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
      generator
      (T5Output decoderOutput encoderOutput inputPaddingMask)
      generatorOutput
  ) =>
  HasForward
    (T5Base hasLMHead device)
    (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward T5Base {..} = forward t5BaseSeqToSeq