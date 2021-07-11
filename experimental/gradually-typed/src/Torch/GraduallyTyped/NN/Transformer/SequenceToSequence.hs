{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.SequenceToSequence where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, SingKind (fromSing), sing)
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat (SNat))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder, TransformerDecoderSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, TransformerEncoderSpec (..))
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead, LMHeadSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead, SWithMLMHead, SWithoutHead), STransformerStyle (..), TransformerHead (WithLMHead, WithoutHead), TransformerStyle (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (), TensorSpec (..))

data
  GSequenceToSequenceTransformer
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoder :: Type)
    (decoder :: Type)
    (embedding :: Type)
    (head :: Type)
  where
  GSequenceToSequenceTransformer ::
    forall inputEmbedDim encoder decoder embedding head.
    { -- | input embedding dim for scaling
      seqToSeqInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      seqToSeqEncoder :: encoder,
      -- | decoder
      seqToSeqDecoder :: decoder,
      -- | shared embedding
      seqToSeqEmbedding :: embedding,
      -- | transformer head
      seqToSeqHead :: head
    } ->
    GSequenceToSequenceTransformer inputEmbedDim encoder decoder embedding head

-- | Sequence-to-sequence transformer model.
newtype
  SequenceToSequenceTransformer
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  SequenceToSequenceTransformer ::
    forall style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim.
    GSequenceToSequenceTransformer
      inputEmbedDim
      (SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim) ->
    SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim

data
  SequenceToSequenceTransformerSpec
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  SequenceToSequenceTransformerSpec ::
    forall style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim.
    STransformerStyle style ->
    STransformerHead transformerHead ->
    SNat numEncoderLayers ->
    SNat numDecoderLayers ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim inputEmbedDim ->
    SDim ffnDim ->
    SDim posEncDim ->
    SDim vocabDim ->
    Double ->
    Double ->
    SequenceToSequenceTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim

type instance ModelSpec (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim) = SequenceToSequenceTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim

type family
  SequenceToSequenceEncoderF
    (style :: TransformerStyle)
    (numEncoderLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
  where
  SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim =
    TransformerEncoder style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim

type family
  SequenceToSequenceDecoderF
    (style :: TransformerStyle)
    (numDecoderLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
  where
  SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim =
    TransformerDecoder style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim

type family
  SequenceToSequenceEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  SequenceToSequenceEmbeddingF _ gradient device dataType inputEmbedDim vocabDim =
    Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing

type family
  SequenceToSequenceHeadF
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SequenceToSequenceHeadF style 'WithoutHead gradient device dataType inputEmbedDim vocabDim =
    ()
  SequenceToSequenceHeadF style 'WithLMHead gradient device dataType inputEmbedDim vocabDim =
    LMHead style gradient device dataType inputEmbedDim vocabDim

instance
  ( encoder ~ TransformerEncoder style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim,
    HasInitialize encoder device encoder device,
    decoder ~ TransformerDecoder style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim,
    HasInitialize decoder device decoder device,
    embedding ~ Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
    HasInitialize embedding device embedding device,
    lmHead ~ SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim,
    HasInitialize lmHead device lmHead device
  ) =>
  HasInitialize
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim)
    generatorDevice
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim)
    device
  where
  initialize (SequenceToSequenceTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        encoder = IxStateT . initialize @encoder $ TransformerEncoderSpec style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        decoder = IxStateT . initialize @decoder $ TransformerDecoderSpec style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        embedding = IxStateT . initialize @embedding $ EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
        lmHead = IxStateT . initialize @lmHead $ case transformerHead of
          SWithoutHead -> ()
          SWithLMHead -> LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps
          SWithMLMHead -> undefined
        gSeqToSeq =
          GSequenceToSequenceTransformer
            <<$>> ireturn inputEmbedDim <<*>> encoder
              <<*>> decoder
              <<*>> embedding
              <<*>> lmHead
     in runIxStateT (gSeqToSeq >>>= ireturn . SequenceToSequenceTransformer) generator'

instance
  (SingI style, SingI transformerHead) =>
  HasStateDict
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim)
  where
  fromStateDict (SequenceToSequenceTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps) k =
    let encoderSpec = TransformerEncoderSpec style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        encoder ST5 = fromStateDict encoderSpec (k <> "encoder.")
        encoder SByT5 = fromStateDict encoderSpec (k <> "encoder.")
        encoder SBART = fromStateDict encoderSpec (k <> "model.encoder.")
        encoder SMBART = fromStateDict encoderSpec (k <> "model.encoder.")
        encoder SPegasus = fromStateDict encoderSpec (k <> "model.encoder.")
        encoder SBERT = undefined
        encoder SRoBERTa = undefined
        encoder SGPT2 = undefined
        decoderSpec = TransformerDecoderSpec style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        decoder ST5 = fromStateDict decoderSpec (k <> "decoder.")
        decoder SByT5 = fromStateDict decoderSpec (k <> "decoder.")
        decoder SBART = fromStateDict decoderSpec (k <> "model.decoder.")
        decoder SMBART = fromStateDict decoderSpec (k <> "model.decoder.")
        decoder SPegasus = fromStateDict decoderSpec (k <> "model.decoder.")
        decoder SBERT = undefined
        decoder SRoBERTa = undefined
        decoder SGPT2 = undefined
        embeddingSpec = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
        embedding ST5 = fromStateDict embeddingSpec (k <> "shared.")
        embedding SByT5 = fromStateDict embeddingSpec (k <> "shared.")
        embedding SBART = fromStateDict embeddingSpec (k <> "model.shared.")
        embedding SMBART = fromStateDict embeddingSpec (k <> "model.shared.")
        embedding SPegasus = fromStateDict embeddingSpec (k <> "model.shared.")
        embedding SBERT = undefined
        embedding SRoBERTa = undefined
        embedding SGPT2 = undefined
        lmHeadSpec = LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps
        lmHead _ SWithoutHead = fromStateDict () k
        lmHead ST5 SWithLMHead = fromStateDict lmHeadSpec (k <> "lm_head.")
        lmHead SByT5 SWithLMHead = fromStateDict lmHeadSpec (k <> "lm_head.")
        lmHead SBART SWithLMHead = fromStateDict lmHeadSpec k
        lmHead SMBART SWithLMHead = fromStateDict lmHeadSpec k
        lmHead SPegasus SWithLMHead = fromStateDict lmHeadSpec k
        lmHead SBERT SWithLMHead = undefined
        lmHead SRoBERTa SWithLMHead = undefined
        lmHead SGPT2 SWithLMHead = undefined
        lmHead _ SWithMLMHead = undefined
     in SequenceToSequenceTransformer
          <$> ( GSequenceToSequenceTransformer
                  inputEmbedDim
                  <$> encoder style
                  <*> decoder style
                  <*> embedding style
                  <*> lmHead style transformerHead
              )
  toStateDict k (SequenceToSequenceTransformer GSequenceToSequenceTransformer {..}) =
    let encoder ST5 = toStateDict (k <> "encoder.")
        encoder SByT5 = toStateDict (k <> "encoder.")
        encoder SBART = toStateDict (k <> "encoder.")
        encoder SMBART = toStateDict (k <> "encoder.")
        encoder SPegasus = toStateDict (k <> "encoder.")
        encoder SBERT = undefined
        encoder SRoBERTa = undefined
        encoder SGPT2 = undefined
        decoder ST5 = toStateDict (k <> "decoder.")
        decoder SByT5 = toStateDict (k <> "decoder.")
        decoder SBART = toStateDict (k <> "decoder.")
        decoder SMBART = toStateDict (k <> "decoder.")
        decoder SPegasus = toStateDict (k <> "decoder.")
        decoder SBERT = undefined
        decoder SRoBERTa = undefined
        decoder SGPT2 = undefined
        embedding ST5 = toStateDict (k <> "shared.")
        embedding SByT5 = toStateDict (k <> "shared.")
        embedding SBART = toStateDict (k <> "shared.")
        embedding SMBART = toStateDict (k <> "shared.")
        embedding SPegasus = toStateDict (k <> "shared.")
        embedding SBERT = undefined
        embedding SRoBERTa = undefined
        embedding SGPT2 = undefined
        lmHead _ SWithoutHead = toStateDict k
        lmHead ST5 SWithLMHead = toStateDict (k <> "lm_head.")
        lmHead SByT5 SWithLMHead = toStateDict (k <> "lm_head.")
        lmHead SBART SWithLMHead = toStateDict k
        lmHead SMBART SWithLMHead = toStateDict k
        lmHead SPegasus SWithLMHead = toStateDict k
        lmHead SBERT SWithLMHead = undefined
        lmHead SRoBERTa SWithLMHead = undefined
        lmHead SGPT2 SWithLMHead = undefined
        lmHead _ SWithMLMHead = undefined
     in do
          () <- encoder (sing @style) seqToSeqEncoder
          () <- decoder (sing @style) seqToSeqDecoder
          () <- embedding (sing @style) seqToSeqEmbedding
          () <- lmHead (sing @style) (sing @transformerHead) seqToSeqHead
          pure ()

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for training.
data SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerInput ::
    forall input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask.
    { input :: input,
      decoderInput :: decoderInput,
      pos :: pos,
      decoderPos :: decoderPos,
      attentionMask :: attentionMask,
      decoderAttentionMask :: decoderAttentionMask,
      crossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask

deriving instance
  ( Show input,
    Show decoderInput,
    Show pos,
    Show decoderPos,
    Show attentionMask,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)

-- | Output data type for use with a sequence-to-sequence transformer.
data SequenceToSequenceTransformerOutput decoderOutput encoderOutput where
  SequenceToSequenceTransformerOutput ::
    forall decoderOutput encoderOutput.
    { decoderOutput :: decoderOutput,
      encoderOutput :: encoderOutput
    } ->
    SequenceToSequenceTransformerOutput decoderOutput encoderOutput

deriving instance
  ( Show decoderOutput,
    Show encoderOutput
  ) =>
  Show (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for inference.
data SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerGenerationInput ::
    forall decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask.
    { generationDecoderInput :: decoderInput,
      generationEncoderOutput :: encoderOutput,
      generationDecoderPos :: decoderPos,
      generationDecoderAttentionMask :: decoderAttentionMask,
      generationCrossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show decoderPos,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)

-- | 'HasForward' instance for sequence-to-sequence transformers without additional head(s).
--
-- @
--     ┌───────┐  ┌─────┐  ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
--     │ input │  │ pos │  │ attentionMask │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
--     └───┬───┘  └──┬──┘  └──────┬────────┘  └──────┬───────┘  └─────┬──────┘  └──────────┬───────────┘  └─────────┬──────────┘
--         │         │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
-- seqToSeqEmbedding │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--   (embedScaling)  │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--  seqToSeqEncoder◄─┘◄───────────┘                  ▼                │                    │                        │
--         │                                 seqToSeqEmbedding        │                    │                        │
--         │                                         ▼                │                    │                        │
--         │                                   (embedScaling)         │                    │                        │
--         │                                         ▼                │                    │                        │
--         ├─────────────────────────────────►seqToSeqDecoder◄────────┘◄───────────────────┘◄───────────────────────┘
--         │                                         │
--         │                                  (seqToSeqLMHead)
--         │                                         │
--         ▼                                         ▼
-- ┌───────────────┐                         ┌───────────────┐
-- │ encoderOutput │                         │ decoderOutput │
-- └───────────────┘                         └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      input
      generatorDevice
      embeddingOutput
      embeddingGeneratorOutputDevice,
    embeddingOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    HasForward
      (SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      (embeddingOutput, pos, attentionMask)
      embeddingGeneratorOutputDevice
      encoderOutput
      encoderGeneratorOutputDevice,
    HasForward
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      decoderInput
      encoderGeneratorOutputDevice
      embeddingOutput'
      embeddingGeneratorOutputDevice',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutputDevice'
      decoderOutput
      generatorOutputDevice
  ) =>
  HasForward
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim)
    (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
    generatorDevice
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutputDevice
  where
  forward (SequenceToSequenceTransformer GSequenceToSequenceTransformer {..}) SequenceToSequenceTransformerInput {..} =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ seqToSeqInputEmbedDim
        embedScaling ::
          forall requiresGradient layout device''' dataType''' shape.
          STransformerStyle style ->
          Tensor requiresGradient layout device''' dataType''' shape ->
          Tensor requiresGradient layout device''' dataType''' shape
        embedScaling ST5 = id
        embedScaling SByT5 = id
        embedScaling SBART = id
        embedScaling SMBART = id
        embedScaling SPegasus = flip mulScalar s
        embedScaling SBERT = undefined
        embedScaling SRoBERTa = undefined
        embedScaling SGPT2 = undefined
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward seqToSeqEmbedding
            >>>= ireturn . embedScaling (sing @style)
            >>>= (\input' -> IxStateT $ forward seqToSeqEncoder (input', pos, attentionMask))
            >>>= ( \encoderOutput ->
                     ireturn decoderInput
                       >>>= IxStateT . forward seqToSeqEmbedding
                       >>>= ireturn . embedScaling (sing @style)
                       >>>= ( \decoderInput' ->
                                IxStateT $ forward seqToSeqDecoder (decoderInput', encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask)
                            )
                       >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
                 )

-- | 'HasForward' instance for sequence-to-sequence transformers without language modelling head.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- @
-- ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ encoderOutput │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └───────┬───────┘  └───────┬──────┘  └──────┬─────┘  └───────────┬──────────┘  └──────────┬─────────┘
--         │                  │                │                    │                        │
--         │                  ▼                │                    │                        │
--         │          seqToSeqEmbedding        │                    │                        │
--         │                  ▼                │                    │                        │
--         │            (embedScaling)         │                    │                        │
--         │                  ▼                │                    │                        │
--         ├──────────►seqToSeqDecoder◄────────┘◄───────────────────┘◄───────────────────────┘
--         │                  │
--         │           (seqToSeqLMHead)
--         │                  │
--         ▼                  ▼
-- ┌───────────────┐  ┌───────────────┐
-- │ encoderOutput │  │ decoderOutput │
-- └───────────────┘  └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      decoderInput
      generatorDevice
      embeddingOutput'
      embeddingGeneratorOutputDevice',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutputDevice'
      decoderOutput
      decoderGeneratorOutputDevice,
    HasForward
      (SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)
      decoderOutput
      decoderGeneratorOutputDevice
      headOutput
      generatorOutputDevice
  ) =>
  HasForward
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
    generatorDevice
    (SequenceToSequenceTransformerOutput headOutput encoderOutput)
    generatorOutputDevice
  where
  forward (SequenceToSequenceTransformer GSequenceToSequenceTransformer {..}) SequenceToSequenceTransformerGenerationInput {..} =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ seqToSeqInputEmbedDim
        embedScaling ::
          forall requiresGradient layout device''' dataType''' shape.
          STransformerStyle style ->
          Tensor requiresGradient layout device''' dataType''' shape ->
          Tensor requiresGradient layout device''' dataType''' shape
        embedScaling ST5 = id
        embedScaling SByT5 = id
        embedScaling SBART = id
        embedScaling SMBART = id
        embedScaling SPegasus = flip mulScalar s
        embedScaling SBERT = undefined
        embedScaling SRoBERTa = undefined
        embedScaling SGPT2 = undefined
     in runIxStateT $
          ireturn generationDecoderInput
            >>>= IxStateT . forward seqToSeqEmbedding
            >>>= ireturn . embedScaling (sing @style)
            >>>= ( \decoderInput' ->
                     IxStateT $ forward seqToSeqDecoder (decoderInput', generationEncoderOutput, generationDecoderPos, generationDecoderAttentionMask, generationCrossAttentionMask)
                 )
            >>>= IxStateT . forward seqToSeqHead
            >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput generationEncoderOutput)

testSeqToSeq :: IO _
testSeqToSeq = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      vocabDim = SName @"*" :&: SSize @32128
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
      attentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      decoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (t5Output, g'') <- do
    (t5, g') <- initialize (SequenceToSequenceTransformerSpec ST5 SWithLMHead (SNat @32) (SNat @32) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps) g
    let pos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
    forward t5 SequenceToSequenceTransformerInput {..} g'
  (bartOutput, g'''') <- do
    (bart, g''') <- initialize (SequenceToSequenceTransformerSpec SBART SWithLMHead (SNat @32) (SNat @32) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps) g''
    let pos = sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
    forward bart SequenceToSequenceTransformerInput {..} g'''
  pure ((t5Output, bartOutput), g'''')
