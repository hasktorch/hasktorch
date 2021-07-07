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
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, SingKind (fromSing), sing)
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder)
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder)
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (SWithLMHead, SWithMLMHead, SWithoutHead), STransformerStyle (..), TransformerHead (WithLMHead, WithoutHead), TransformerStyle (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor ())

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
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformer ::
    forall style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP.
    GSequenceToSequenceTransformer
      inputEmbedDim
      (SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim) ->
    SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

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
    (dropoutP :: Type)
  where
  SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP =
    TransformerEncoder style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

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
    (dropoutP :: Type)
  where
  SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP =
    TransformerDecoder style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP

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

type family
  HasInitializeSequenceToSequenceHeadInputF
    (transformerHead :: TransformerHead)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeSequenceToSequenceHeadInputF 'WithoutHead _ _ _ _ _ = ()
  HasInitializeSequenceToSequenceHeadInputF 'WithLMHead gradient device dataType inputEmbedDim vocabDim =
    (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim, Double)

instance
  ( SingI transformerHead,
    HasInitialize
      (TransformerEncoder style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
      generator
      generator',
    HasInitialize
      (TransformerDecoder style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
      generator'
      generator'',
    HasInitialize
      (Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim vocabDim, SDim inputEmbedDim)
      generator''
      generator''',
    HasInitialize
      (SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)
      (HasInitializeSequenceToSequenceHeadInputF transformerHead gradient device dataType inputEmbedDim vocabDim)
      generator'''
      generator''''
  ) =>
  HasInitialize
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, SDim vocabDim, dropoutP, Double)
    generator
    generator''''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, dropoutP, eps) =
    let encoder = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps)
        decoder = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps)
        embedding = IxState . initialize $ (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim)
        head = IxState . initialize $ case sing @transformerHead of
          SWithoutHead -> ()
          SWithLMHead -> (gradient, device, dataType, inputEmbedDim, vocabDim, eps)
          SWithMLMHead -> undefined
     in runIxState $
          ( GSequenceToSequenceTransformer
              <<$>> ireturn inputEmbedDim <<*>> encoder
                <<*>> decoder
                <<*>> embedding
                <<*>> head
          )
            >>>= ireturn . SequenceToSequenceTransformer

instance
  (SingI style, SingI transformerHead, KnownNat numEncoderLayers, KnownNat numDecoderLayers) =>
  HasStateDict
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, SDim vocabDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, dropoutP, eps) k =
    let encoder ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "encoder.")
        encoder SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "encoder.")
        encoder SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.encoder.")
        encoder SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.encoder.")
        encoder SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.encoder.")
        encoder SBERT = undefined
        encoder SRoBERTa = undefined
        encoder SGPT2 = undefined
        decoder ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "decoder.")
        decoder SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "decoder.")
        decoder SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.decoder.")
        decoder SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.decoder.")
        decoder SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) (k <> "model.decoder.")
        decoder SBERT = undefined
        decoder SRoBERTa = undefined
        decoder SGPT2 = undefined
        embedding ST5 = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "shared.")
        embedding SByT5 = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "shared.")
        embedding SBART = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "model.shared.")
        embedding SMBART = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "model.shared.")
        embedding SPegasus = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "model.shared.")
        embedding SBERT = undefined
        embedding SRoBERTa = undefined
        embedding SGPT2 = undefined
        lmHead _ SWithoutHead = fromStateDict () k
        lmHead ST5 SWithLMHead = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) (k <> "lm_head.")
        lmHead SByT5 SWithLMHead = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) (k <> "lm_head.")
        lmHead SBART SWithLMHead = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) k
        lmHead SMBART SWithLMHead = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) k
        lmHead SPegasus SWithLMHead = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) k
        lmHead SBERT SWithLMHead = undefined
        lmHead SRoBERTa SWithLMHead = undefined
        lmHead SGPT2 SWithLMHead = undefined
        lmHead _ SWithMLMHead = undefined
     in SequenceToSequenceTransformer
          <$> ( GSequenceToSequenceTransformer
                  inputEmbedDim
                  <$> encoder (sing @style)
                  <*> decoder (sing @style)
                  <*> embedding (sing @style)
                  <*> lmHead (sing @style) (sing @transformerHead)
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
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    embeddingOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    HasForward
      (SequenceToSequenceEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (embeddingOutput, pos, attentionMask)
      embeddingGeneratorOutput
      encoderOutput
      encoderGeneratorOutput,
    HasForward
      (SequenceToSequenceEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      decoderInput
      encoderGeneratorOutput
      embeddingOutput'
      embeddingGeneratorOutput',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
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
      generator
      embeddingOutput'
      embeddingGeneratorOutput',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SequenceToSequenceDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      decoderGeneratorOutput,
    HasForward
      (SequenceToSequenceHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)
      decoderOutput
      decoderGeneratorOutput
      headOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput headOutput encoderOutput)
    generatorOutput
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
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
      attentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      decoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  (t5Output, g'') <-
    let (t5, g') = initialize @(SequenceToSequenceTransformer 'T5 'WithLMHead 32 32 _ _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, dropoutP, eps) g
        pos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
     in forward t5 SequenceToSequenceTransformerInput {..} g'
  (bartOutput, g'''') <-
    let (bart, g''') = initialize @(SequenceToSequenceTransformer 'BART 'WithLMHead 32 32 _ _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, dropoutP, eps) g''
        pos = sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
     in forward bart SequenceToSequenceTransformerInput {..} g'''
  pure ((t5Output, bartOutput), g'''')
