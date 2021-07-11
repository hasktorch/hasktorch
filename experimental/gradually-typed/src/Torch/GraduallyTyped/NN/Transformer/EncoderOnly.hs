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
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, SingKind (fromSing), sing)
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, TransformerEncoderSpec (..))
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead, LMHeadSpec (..))
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
  where
  EncoderOnlyTransformer ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    GEncoderOnlyTransformer
      inputEmbedDim
      (EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      (EOEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (EOTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim) ->
    EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

data
  EncoderOnlyTransformerSpec
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
  where
  EncoderOnlyTransformerSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    STransformerStyle style ->
    SNat numLayers ->
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
    SDim typeVocabDim ->
    Double ->
    Double ->
    EncoderOnlyTransformerSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

type instance ModelSpec (EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim) = EncoderOnlyTransformerSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

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
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim =
    TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim

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
  where
  EncoderOnlyTransformerWithLMHead ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    GEncoderOnlyTransformerWithLMHead
      (EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
      (EOLMHeadF style gradient device dataType inputEmbedDim vocabDim) ->
    EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

data
  EncoderOnlyTransformerWithLMHeadSpec
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
  where
  EncoderOnlyTransformerWithLMHeadSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    STransformerStyle style ->
    SNat numLayers ->
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
    SDim typeVocabDim ->
    Double ->
    Double ->
    EncoderOnlyTransformerWithLMHeadSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

type instance ModelSpec (EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim) = EncoderOnlyTransformerWithLMHeadSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

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
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim =
    EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

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
  SingI style =>
  HasStateDict
    (EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
  where
  fromStateDict (EncoderOnlyTransformerSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP eps) k =
    let encoderSpec = TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        encoder ST5 = undefined
        encoder SByT5 = undefined
        encoder SBART = undefined
        encoder SMBART = undefined
        encoder SPegasus = undefined
        encoder SBERT = fromStateDict encoderSpec k
        encoder SRoBERTa = fromStateDict encoderSpec k
        encoder SGPT2 = undefined
        embeddingSpec = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
        embedding ST5 = undefined
        embedding SByT5 = undefined
        embedding SBART = undefined
        embedding SMBART = undefined
        embedding SPegasus = undefined
        embedding SBERT = fromStateDict embeddingSpec (k <> "embeddings.word_embeddings.")
        embedding SRoBERTa = fromStateDict embeddingSpec (k <> "embeddings.word_embeddings.")
        embedding SGPT2 = undefined
        typeEmbeddingSpec = EmbeddingSpec gradient (SLayout SDense) device dataType typeVocabDim inputEmbedDim SNothing
        typeEmbedding ST5 = undefined
        typeEmbedding SByT5 = undefined
        typeEmbedding SBART = undefined
        typeEmbedding SMBART = undefined
        typeEmbedding SPegasus = undefined
        typeEmbedding SBERT = fromStateDict typeEmbeddingSpec (k <> "embeddings.token_type_embeddings.")
        typeEmbedding SRoBERTa = fromStateDict typeEmbeddingSpec (k <> "embeddings.token_type_embeddings.")
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
  SingI style =>
  HasStateDict
    (EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
  where
  fromStateDict (EncoderOnlyTransformerWithLMHeadSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP eps) k =
    let transformerSpec = EncoderOnlyTransformerSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP eps
        transformer ST5 = undefined
        transformer SByT5 = undefined
        transformer SBART = undefined
        transformer SMBART = undefined
        transformer SPegasus = undefined
        transformer SBERT = fromStateDict transformerSpec (k <> "bert.")
        transformer SRoBERTa = fromStateDict transformerSpec (k <> "roberta.")
        transformer SGPT2 = undefined
        lmHeadSpec = LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps
        lmHead ST5 = undefined
        lmHead SByT5 = undefined
        lmHead SBART = undefined
        lmHead SMBART = undefined
        lmHead SPegasus = undefined
        lmHead SBERT = fromStateDict lmHeadSpec (k <> "cls.predictions.")
        lmHead SRoBERTa = fromStateDict lmHeadSpec (k <> "lm_head.")
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
      generatorDevice
      embeddingOutput
      embeddingGeneratorOutputDevice,
    embeddingOutput ~ Tensor gradient' layout' device' dataType' shape',
    HasForward
      (EOTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim)
      inputType
      embeddingGeneratorOutputDevice
      typeEmbeddingOutput
      typeEmbeddingGeneratorOutputDevice,
    typeEmbeddingOutput ~ Tensor gradient'' layout'' device'' dataType'' shape'',
    HasForward
      (EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      ( Tensor
          (gradient' <|> gradient'')
          (layout' <+> layout'')
          (device' <+> device'')
          (dataType' <+> dataType'')
          (BroadcastShapesF shape' shape''),
        pos,
        attentionMask
      )
      typeEmbeddingGeneratorOutputDevice
      encoderOutput
      generatorOutputDevice
  ) =>
  HasForward
    (EncoderOnlyTransformer style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
    (EncoderOnlyTransformerInput input inputType pos attentionMask)
    generatorDevice
    (EncoderOnlyTransformerOutput encoderOutput)
    generatorOutputDevice
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
      (EOTransformerF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
      input
      generatorDevice
      eoOutput
      eoGeneratorOutputDevice,
    eoOutput ~ EncoderOnlyTransformerOutput encoderOutput,
    HasForward
      (EOLMHeadF style gradient device dataType inputEmbedDim vocabDim)
      encoderOutput
      eoGeneratorOutputDevice
      lmHeadOutput
      generatorOutputDevice,
    output ~ EncoderOnlyTransformerOutput lmHeadOutput
  ) =>
  HasForward
    (EncoderOnlyTransformerWithLMHead style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
    input
    generatorDevice
    output
    generatorOutputDevice
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