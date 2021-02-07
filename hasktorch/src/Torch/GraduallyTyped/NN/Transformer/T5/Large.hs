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
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5Input, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformerWithoutLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Large number of layers.
-- 'num_layers = 24'
type T5LargeNumLayers = 24

-- | T5-Large number of attention heads.
-- 'n_heads = 16'
type T5LargeHeadDim = 'Dim ( 'Name "*") ( 'Size 16)

-- | T5-Large head embedding dimension.
-- 'd_kv = 64'
type T5LargeHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | T5-Large embedding dimension.
-- 'inner_dim = n_heads * d_kv = 1024'
type T5LargeEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

-- | T5-Large model dimension.
-- 'd_model = 1024'
type T5LargeInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

-- | T5-Large feed-forward network dimension.
-- 'd_ff = 4096'
type T5LargeFFNDim = 'Dim ( 'Name "*") ( 'Size 4096)

-- | T5-Large vocabulary dimension.
-- 'vocab_size = 32128'
type T5LargeVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | T5-Large configuration data type.
-- Modelled after https://huggingface.co/t5-large/blob/main/config.json.
type T5LargeConfig device =
  T5Config T5LargeNumLayers device T5LargeHeadDim T5LargeHeadEmbedDim T5LargeEmbedDim T5LargeInputEmbedDim T5LargeFFNDim T5RelPosEncBucketDim T5LargeVocabDim

-- | load a T5-Large configuration from a file
t5LargeConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5LargeConfig ( 'Device 'CPU))
t5LargeConfigFromPretrained = t5ConfigFromPretrained

-- | T5-Large data type.
data
  T5Large
    (hasLMHead :: HasLMHead)
    (device :: Device (DeviceType Nat))
  where
  -- | T5-Large constructor.
  T5Large ::
    forall hasLMHead device.
    { t5LargeSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5LargeNumLayers
          T5LargeNumLayers
          device
          T5DataType
          T5LargeHeadDim
          T5LargeHeadEmbedDim
          T5LargeEmbedDim
          T5LargeInputEmbedDim
          T5LargeFFNDim
          T5RelPosEncBucketDim
          T5LargeVocabDim
          T5DropoutP
    } ->
    T5Large hasLMHead device

instance HasInitialize (T5Large 'WithoutLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5Large 'WithoutLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5Large 'WithoutLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5LargeConfigFromPretrained filePath False
    flip runReaderT config $
      T5Large <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5Large 'WithLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5Large 'WithLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5Large 'WithLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5LargeConfigFromPretrained filePath False
    flip runReaderT config $
      T5Large <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5LargeNumLayers
        T5LargeNumLayers
        device
        T5DataType
        T5LargeHeadDim
        T5LargeHeadEmbedDim
        T5LargeEmbedDim
        T5LargeInputEmbedDim
        T5LargeFFNDim
        T5RelPosEncBucketDim
        T5LargeVocabDim
        T5DropoutP
    )
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput =>
  HasForward
    (T5Large hasLMHead device)
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput
  where
  forward T5Large {..} = forward t5LargeSeqToSeq

instance
  ( HasForward
      ( SequenceToSequenceTransformer
          hasLMHead
          T5LargeNumLayers
          T5LargeNumLayers
          device
          T5DataType
          T5LargeHeadDim
          T5LargeHeadEmbedDim
          T5LargeEmbedDim
          T5LargeInputEmbedDim
          T5LargeFFNDim
          T5RelPosEncBucketDim
          T5LargeVocabDim
          T5DropoutP
      )
      (T5Input input decoderInput)
      generator
      output
      generatorOutput
  ) =>
  HasForward
    (T5Large hasLMHead device)
    (T5Input input decoderInput)
    generator
    output
    generatorOutput
  where
  forward T5Large {..} = forward t5LargeSeqToSeq
