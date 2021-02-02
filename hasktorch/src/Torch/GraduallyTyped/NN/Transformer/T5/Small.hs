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
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer)
import Torch.GraduallyTyped.NN.Transformer.T5.Common (T5Config, T5DataType, T5DropoutP, T5RelPosEncBucketDim, lookupSequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformerWithoutLMHead, t5ConfigFromPretrained)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Size (..))

-- | T5-Small number of layers.
-- 'num_layers = 6'
type T5SmallNumLayers = 6

-- | T5-Small number of attention heads.
-- 'n_heads = 8'
type T5SmallHeadDim = 'Dim ( 'Name "*") ( 'Size 8)

-- | T5-Small head embedding dimension.
-- 'd_kv = 64'
type T5SmallHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | T5-Small embedding dimension.
-- 'inner_dim = n_heads * d_kv = 512'
type T5SmallEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | T5-Small model dimension.
-- 'd_model = 512'
type T5SmallInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | T5-Small feed-forward network dimension.
-- 'd_ff = 2048'
type T5SmallFFNDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- | T5-Small vocabulary dimension.
-- 'vocab_size = 32128'
type T5SmallVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | T5-Small configuration data type.
-- Modelled after https://huggingface.co/t5-small/blob/main/config.json.
type T5SmallConfig device =
  T5Config T5SmallNumLayers device T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5RelPosEncBucketDim T5SmallVocabDim

-- | T5-Small data type.
data
  T5Small
    (hasLMHead :: HasLMHead)
    (device :: Device (DeviceType Nat))
  where
  -- | T5-Small constructor.
  T5Small ::
    forall hasLMHead device.
    { t5SmallSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5SmallNumLayers
          T5SmallNumLayers
          device
          T5DataType
          T5SmallHeadDim
          T5SmallHeadEmbedDim
          T5SmallEmbedDim
          T5SmallInputEmbedDim
          T5SmallFFNDim
          T5RelPosEncBucketDim
          T5SmallVocabDim
          T5DropoutP
    } ->
    T5Small hasLMHead device

-- | load a T5-Small configuration from a file
t5SmallConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5SmallConfig ( 'Device 'CPU))
t5SmallConfigFromPretrained = t5ConfigFromPretrained

instance HasInitialize (T5Small 'WithoutLMHead device) where
  type
    InitializeF (T5Small 'WithoutLMHead device) =
      T5SmallConfig device -> IO (T5Small 'WithoutLMHead device)
  initialize config =
    flip runReaderT config $
      T5Small <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5Small 'WithLMHead device) where
  type
    InitializeF (T5Small 'WithLMHead device) =
      T5SmallConfig device -> IO (T5Small 'WithLMHead device)
  initialize config =
    flip runReaderT config $
      T5Small <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5SmallNumLayers
        T5SmallNumLayers
        device
        T5DataType
        T5SmallHeadDim
        T5SmallHeadEmbedDim
        T5SmallEmbedDim
        T5SmallInputEmbedDim
        T5SmallFFNDim
        T5RelPosEncBucketDim
        T5SmallVocabDim
        T5DropoutP
    )
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput =>
  HasForward
    (T5Small hasLMHead device)
    (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
    generator
    output
    generatorOutput
  where
  forward T5Small {..} = forward t5SmallSeqToSeq

instance
  ( HasForward
      ( SequenceToSequenceTransformer
          hasLMHead
          T5SmallNumLayers
          T5SmallNumLayers
          device
          T5DataType
          T5SmallHeadDim
          T5SmallHeadEmbedDim
          T5SmallEmbedDim
          T5SmallInputEmbedDim
          T5SmallFFNDim
          T5RelPosEncBucketDim
          T5SmallVocabDim
          T5DropoutP
      )
      ( input,
        decoderInput
      )
      generator
      output
      generatorOutput
  ) =>
  HasForward
    (T5Small hasLMHead device)
    (input, decoderInput)
    generator
    output
    generatorOutput
  where
  forward T5Small {..} = forward t5SmallSeqToSeq
