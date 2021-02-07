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
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5Input, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformerWithoutLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-3B number of layers.
-- 'num_layers = 24'
type T5ThreeBNumLayers = 24

-- | T5-3B number of attention heads.
-- 'n_heads = 32'
type T5ThreeBHeadDim = 'Dim ( 'Name "*") ( 'Size 32)

-- | T5-3B head embedding dimension.
-- 'd_kv = 128'
type T5ThreeBHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 128)

-- | T5-3B embedding dimension.
-- 'inner_dim = n_heads * d_kv = 4096'
type T5ThreeBEmbedDim = 'Dim ( 'Name "*") ( 'Size 4096)

-- | T5-3B model dimension.
-- 'd_model = 1024'
type T5ThreeBInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 1024)

-- | T5-3B feed-forward network dimension.
-- 'd_ff = 16384'
type T5ThreeBFFNDim = 'Dim ( 'Name "*") ( 'Size 16384)

-- | T5-3B vocabulary dimension.
-- 'vocab_size = 32128'
type T5ThreeBVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | T5-3B configuration data type.
-- Modelled after https://huggingface.co/t5-3b/blob/main/config.json.
type T5ThreeBConfig device =
  T5Config T5ThreeBNumLayers device T5ThreeBHeadDim T5ThreeBHeadEmbedDim T5ThreeBEmbedDim T5ThreeBInputEmbedDim T5ThreeBFFNDim T5RelPosEncBucketDim T5ThreeBVocabDim

-- | load a T5-3B configuration from a file
t5ThreeBConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5ThreeBConfig ( 'Device 'CPU))
t5ThreeBConfigFromPretrained = t5ConfigFromPretrained

-- | T5-3B data type.
data
  T5ThreeB
    (hasLMHead :: HasLMHead)
    (device :: Device (DeviceType Nat))
  where
  -- | T5-3B constructor.
  T5ThreeB ::
    forall hasLMHead device.
    { t5ThreeBSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5ThreeBNumLayers
          T5ThreeBNumLayers
          device
          T5DataType
          T5ThreeBHeadDim
          T5ThreeBHeadEmbedDim
          T5ThreeBEmbedDim
          T5ThreeBInputEmbedDim
          T5ThreeBFFNDim
          T5RelPosEncBucketDim
          T5ThreeBVocabDim
          T5DropoutP
    } ->
    T5ThreeB hasLMHead device

instance HasInitialize (T5ThreeB 'WithoutLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5ThreeB 'WithoutLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5ThreeB 'WithoutLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5ThreeBConfigFromPretrained filePath False
    flip runReaderT config $
      T5ThreeB <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5ThreeB 'WithLMHead ( 'Device 'CPU)) where
  type
    InitializeF (T5ThreeB 'WithLMHead ( 'Device 'CPU)) =
      FilePath -> IO (T5ThreeB 'WithLMHead ( 'Device 'CPU))
  initialize filePath = do
    config <- t5ThreeBConfigFromPretrained filePath False
    flip runReaderT config $
      T5ThreeB <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5ThreeBNumLayers
        T5ThreeBNumLayers
        device
        T5DataType
        T5ThreeBHeadDim
        T5ThreeBHeadEmbedDim
        T5ThreeBEmbedDim
        T5ThreeBInputEmbedDim
        T5ThreeBFFNDim
        T5RelPosEncBucketDim
        T5ThreeBVocabDim
        T5DropoutP
    )
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput =>
  HasForward
    (T5ThreeB hasLMHead device)
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput
  where
  forward T5ThreeB {..} = forward t5ThreeBSeqToSeq

instance
  ( HasForward
      ( SequenceToSequenceTransformer
          hasLMHead
          T5ThreeBNumLayers
          T5ThreeBNumLayers
          device
          T5DataType
          T5ThreeBHeadDim
          T5ThreeBHeadEmbedDim
          T5ThreeBEmbedDim
          T5ThreeBInputEmbedDim
          T5ThreeBFFNDim
          T5RelPosEncBucketDim
          T5ThreeBVocabDim
          T5DropoutP
      )
      (T5Input input decoderInput)
      generator
      output
      generatorOutput
  ) =>
  HasForward
    (T5ThreeB hasLMHead device)
    (T5Input input decoderInput)
    generator
    output
    generatorOutput
  where
  forward T5ThreeB {..} = forward t5ThreeBSeqToSeq
