{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.EncoderOnly where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, SingKind (fromSing), sing)
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder)
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack ()
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic encoder-only transformer model.
data
  GEncoderOnlyTransformer
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoder :: Type)
    (encoderEmbedding :: Type)
    (encoderTypeEmbedding :: Type)
  where
  GEncoderOnlyTransformer ::
    forall inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding.
    { -- | input embedding dim for scaling
      eoInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      eoEncoder :: encoder,
      -- | encoder embedding
      eoEmbedding :: encoderEmbedding,
      -- | encoder type embedding
      eoTypeEmbedding :: encoderTypeEmbedding
    } ->
    GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding

-- | Encoder-only transformer model.
data
  EncoderOnlyTransformer
    (style :: TransformerStyle)
    (numLayers :: Nat)
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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  EncoderOnlyTransformer ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP.
    GEncoderOnlyTransformer
      inputEmbedDim
      (EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (EOEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (EOTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim) ->
    EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type family
  EOEncoderF
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP =
    TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type family
  EOEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOEmbeddingF _ gradient device dataType inputEmbedDim vocabDim =
    Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing

type family
  EOTypeEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTypeEmbeddingF _ gradient device dataType inputEmbedDim typeVocabDim =
    Embedding gradient ('Layout 'Dense) device dataType typeVocabDim inputEmbedDim 'Nothing

data
  GEncoderOnlyTransformerWithLMHead
    (transformer :: Type)
    (lmHead :: Type)
  where
  GEncoderOnlyTransformerWithLMHead ::
    forall transformer lmHead.
    { -- | encoder-only transformer
      eoTransformer :: transformer,
      -- | language modelling head
      eoLMHead :: lmHead
    } ->
    GEncoderOnlyTransformerWithLMHead transformer lmHead

-- | Encoder-only transformer model with language modelling head.
data
  EncoderOnlyTransformerWithLMHead
    (style :: TransformerStyle)
    (numLayers :: Nat)
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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  EncoderOnlyTransformerWithLMHead ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP.
    GEncoderOnlyTransformerWithLMHead
      (EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
      (EOLMHeadF style gradient device dataType inputEmbedDim vocabDim) ->
    EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type family
  EOTransformerF
    (style :: TransformerStyle)
    (numLayers :: Nat)
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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP =
    EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type family
  EOLMHeadF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOLMHeadF style gradient device dataType inputEmbedDim vocabDim = LMHead style gradient device dataType inputEmbedDim vocabDim

instance
  (SingI style, KnownNat numLayers) =>
  HasStateDict
    (EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, SDim vocabDim, SDim typeVocabDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, typeVocabDim, dropoutP, eps) k =
    let encoder ST5 = undefined
        encoder SByT5 = undefined
        encoder SBART = undefined
        encoder SMBART = undefined
        encoder SPegasus = undefined
        encoder SBERT = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) k
        encoder SRoBERTa = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) k
        encoder SGPT2 = undefined
        embedding ST5 = undefined
        embedding SByT5 = undefined
        embedding SBART = undefined
        embedding SMBART = undefined
        embedding SPegasus = undefined
        embedding SBERT = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "embeddings.word_embeddings.")
        embedding SRoBERTa = fromStateDict (gradient, SLayout SDense, device, dataType, vocabDim, inputEmbedDim) (k <> "embeddings.word_embeddings.")
        embedding SGPT2 = undefined
        typeEmbedding ST5 = undefined
        typeEmbedding SByT5 = undefined
        typeEmbedding SBART = undefined
        typeEmbedding SMBART = undefined
        typeEmbedding SPegasus = undefined
        typeEmbedding SBERT = fromStateDict (gradient, SLayout SDense, device, dataType, typeVocabDim, inputEmbedDim) (k <> "embeddings.token_type_embeddings.")
        typeEmbedding SRoBERTa = fromStateDict (gradient, SLayout SDense, device, dataType, typeVocabDim, inputEmbedDim) (k <> "embeddings.token_type_embeddings.")
        typeEmbedding SGPT2 = undefined
     in EncoderOnlyTransformer
          <$> ( GEncoderOnlyTransformer
                  inputEmbedDim
                  <$> encoder (sing @style)
                  <*> embedding (sing @style)
                  <*> typeEmbedding (sing @style)
              )
  toStateDict k (EncoderOnlyTransformer GEncoderOnlyTransformer {..}) =
    let encoder ST5 = undefined
        encoder SByT5 = undefined
        encoder SBART = undefined
        encoder SMBART = undefined
        encoder SPegasus = undefined
        encoder SBERT = toStateDict k
        encoder SRoBERTa = toStateDict k
        encoder SGPT2 = undefined
        embedding ST5 = undefined
        embedding SByT5 = undefined
        embedding SBART = undefined
        embedding SMBART = undefined
        embedding SPegasus = undefined
        embedding SBERT = toStateDict (k <> "embeddings.word_embeddings.")
        embedding SRoBERTa = toStateDict (k <> "embeddings.word_embeddings.")
        embedding SGPT2 = undefined
        typeEmbedding ST5 = undefined
        typeEmbedding SByT5 = undefined
        typeEmbedding SBART = undefined
        typeEmbedding SMBART = undefined
        typeEmbedding SPegasus = undefined
        typeEmbedding SBERT = toStateDict (k <> "embeddings.token_type_embeddings.")
        typeEmbedding SRoBERTa = toStateDict (k <> "embeddings.token_type_embeddings.")
        typeEmbedding SGPT2 = undefined
     in do
          () <- encoder (sing @style) eoEncoder
          () <- embedding (sing @style) eoEmbedding
          () <- typeEmbedding (sing @style) eoTypeEmbedding
          pure ()

instance
  (SingI style, KnownNat numLayers) =>
  HasStateDict
    (EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, SDim vocabDim, SDim typeVocabDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, typeVocabDim, dropoutP, eps) k =
    let transformer ST5 = undefined
        transformer SByT5 = undefined
        transformer SBART = undefined
        transformer SMBART = undefined
        transformer SPegasus = undefined
        transformer SBERT = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, typeVocabDim, dropoutP, eps) (k <> "bert.")
        transformer SRoBERTa = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, vocabDim, typeVocabDim, dropoutP, eps) (k <> "roberta.")
        transformer SGPT2 = undefined
        lmHead ST5 = undefined
        lmHead SByT5 = undefined
        lmHead SBART = undefined
        lmHead SMBART = undefined
        lmHead SPegasus = undefined
        lmHead SBERT = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) (k <> "cls.predictions.")
        lmHead SRoBERTa = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) (k <> "lm_head.")
        lmHead SGPT2 = undefined
     in EncoderOnlyTransformerWithLMHead
          <$> ( GEncoderOnlyTransformerWithLMHead
                  <$> transformer (sing @style)
                  <*> lmHead (sing @style)
              )
  toStateDict k (EncoderOnlyTransformerWithLMHead GEncoderOnlyTransformerWithLMHead {..}) =
    let transformer ST5 = undefined
        transformer SByT5 = undefined
        transformer SBART = undefined
        transformer SMBART = undefined
        transformer SPegasus = undefined
        transformer SBERT = toStateDict (k <> "bert.")
        transformer SRoBERTa = toStateDict (k <> "roberta.")
        transformer SGPT2 = undefined
        lmHead ST5 = undefined
        lmHead SByT5 = undefined
        lmHead SBART = undefined
        lmHead SMBART = undefined
        lmHead SPegasus = undefined
        lmHead SBERT = toStateDict (k <> "cls.predictions.")
        lmHead SRoBERTa = toStateDict (k <> "lm_head.")
        lmHead SGPT2 = undefined
     in do
          () <- transformer (sing @style) eoTransformer
          () <- lmHead (sing @style) eoLMHead
          pure ()

-- | Input data type for use with an encoder-only transformer.
data EncoderOnlyTransformerInput input inputType pos attentionMask where
  EncoderOnlyTransformerInput ::
    forall input inputType pos attentionMask.
    { eoInput :: input,
      eoInputType :: inputType,
      eoPos :: pos,
      eoAttentionMask :: attentionMask
    } ->
    EncoderOnlyTransformerInput input inputType pos attentionMask

deriving instance
  ( Show input,
    Show inputType,
    Show pos,
    Show attentionMask
  ) =>
  Show (EncoderOnlyTransformerInput input inputType pos attentionMask)

-- | Output data type for use with an encoder-only transformer.
data EncoderOnlyTransformerOutput encoderOutput where
  EncoderOnlyTransformerOutput ::
    forall encoderOutput.
    { eoEncoderOutput :: encoderOutput
    } ->
    EncoderOnlyTransformerOutput encoderOutput

deriving instance
  ( Show encoderOutput
  ) =>
  Show (EncoderOnlyTransformerOutput encoderOutput)

-- | 'HasForward' instance for encoder-only transformers without additional head(s).
--
-- @
--     ┌───────┐  ┌─────┐  ┌───────────────┐
--     │ input │  │ pos │  │ attentionMask │
--     └───┬───┘  └──┬──┘  └──────┬────────┘
--         │         │            │
--         ▼         │            │
-- seqToSeqEmbedding │            │
--         ▼         │            │
--   (embedScaling)  │            │
--         ▼         │            │
--  seqToSeqEncoder◄─┘◄───────────┘
--         │
--         ▼
-- ┌───────────────┐
-- │ encoderOutput │
-- └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (EOEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    embeddingOutput ~ Tensor gradient' layout' device' dataType' shape',
    HasForward
      (EOTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim)
      inputType
      embeddingGeneratorOutput
      typeEmbeddingOutput
      typeEmbeddingGeneratorOutput,
    typeEmbeddingOutput ~ Tensor gradient'' layout'' device'' dataType'' shape'',
    HasForward
      (EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( Tensor
          (gradient' <|> gradient'')
          (layout' <+> layout'')
          (device' <+> device'')
          (dataType' <+> dataType'')
          (BroadcastShapesF shape' shape''),
        pos,
        attentionMask
      )
      typeEmbeddingGeneratorOutput
      encoderOutput
      generatorOutput
  ) =>
  HasForward
    (EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    (EncoderOnlyTransformerInput input inputType pos attentionMask)
    generator
    (EncoderOnlyTransformerOutput encoderOutput)
    generatorOutput
  where
  forward (EncoderOnlyTransformer GEncoderOnlyTransformer {..}) EncoderOnlyTransformerInput {..} =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ eoInputEmbedDim
        embedScaling ::
          forall gradient''' layout device''' dataType''' shape.
          STransformerStyle style ->
          Tensor gradient''' layout device''' dataType''' shape ->
          Tensor gradient''' layout device''' dataType''' shape
        embedScaling SBERT = id
        embedScaling SRoBERTa = id
        -- embedScaling _ = flip mulScalar s
        embeddedInput =
          ireturn eoInput
            >>>= IxStateT . forward eoEmbedding
            >>>= ireturn . embedScaling (sing @style)
        embeddedInputType =
          ireturn eoInputType
            >>>= IxStateT . forward eoTypeEmbedding
            >>>= ireturn . embedScaling (sing @style)
     in runIxStateT $
          add <<$>> embeddedInput <<*>> embeddedInputType
            >>>= (\input' -> IxStateT $ forward eoEncoder (input', eoPos, eoAttentionMask))
            >>>= ireturn . EncoderOnlyTransformerOutput

-- | 'HasForward' instance for encoder-only transformers with language modelling head.
--
-- @
--     ┌───────┐
--     │ input │
--     └───┬───┘
--         │
--         ▼
--   eoTransformer
--         ▼
--     eoLMHead
--         │
--         ▼
-- ┌───────────────┐
-- │ decoderOutput │
-- └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
      input
      generator
      eoOutput
      eoGeneratorOutput,
    eoOutput ~ EncoderOnlyTransformerOutput encoderOutput,
    HasForward
      (EOLMHeadF style gradient device dataType inputEmbedDim vocabDim)
      encoderOutput
      eoGeneratorOutput
      lmHeadOutput
      generatorOutput,
    output ~ EncoderOnlyTransformerOutput lmHeadOutput
  ) =>
  HasForward
    (EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    input
    generator
    output
    generatorOutput
  where
  forward (EncoderOnlyTransformerWithLMHead GEncoderOnlyTransformerWithLMHead {..}) input =
    runIxStateT $
      ireturn input
        >>>= IxStateT . forward eoTransformer
        >>>= ( \EncoderOnlyTransformerOutput {..} ->
                 ireturn eoEncoderOutput
                   >>>= IxStateT . forward eoLMHead
                   >>>= \lmHeadOutput -> ireturn (EncoderOnlyTransformerOutput lmHeadOutput)
             )