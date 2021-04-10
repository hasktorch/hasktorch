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
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.EncoderOnly where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, lookupEncoder)
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead, lookupLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic encoder-only transformer model.
data
  GEncoderOnlyTransformer
    (encoder :: Type)
    (encoderEmbedding :: Type)
    (encoderTypeEmbedding :: Type)
  where
  GEncoderOnlyTransformer ::
    forall encoder encoderEmbedding encoderTypeEmbedding.
    { -- | encoder
      eoEncoder :: encoder,
      -- | encoder embedding
      eoEmbedding :: encoderEmbedding,
      -- | encoder type embedding
      eoTypeEmbedding :: encoderTypeEmbedding,
      -- | input embedding dim for scaling
      eoInputEmbedDim :: Dim String Integer
    } ->
    GEncoderOnlyTransformer encoder encoderEmbedding encoderTypeEmbedding

-- | Encoder-only transformer model.
data
  EncoderOnlyTransformer
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
    forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP.
    GEncoderOnlyTransformerF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP ->
    EncoderOnlyTransformer numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type GEncoderOnlyTransformerF
  (numLayers :: Nat)
  (style :: TransformerStyle)
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
  (dropoutP :: Type) =
  GEncoderOnlyTransformer
    (EOEncoderF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (EOEmbeddingF style device dataType inputEmbedDim vocabDim)
    (EOTypeEmbeddingF style device dataType inputEmbedDim typeVocabDim)

type family
  EOEncoderF
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
  EOEncoderF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP = TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type family
  EOEmbeddingF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOEmbeddingF _ device dataType inputEmbedDim vocabDim = Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing

type family
  EOTypeEmbeddingF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTypeEmbeddingF _ device dataType inputEmbedDim typeVocabDim = Embedding ('Layout 'Dense) device dataType typeVocabDim inputEmbedDim 'Nothing

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
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
    forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP.
    GEncoderOnlyTransformerWithLMHeadF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP ->
    EncoderOnlyTransformerWithLMHead numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type GEncoderOnlyTransformerWithLMHeadF
  (numLayers :: Nat)
  (style :: TransformerStyle)
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
  (dropoutP :: Type) =
  GEncoderOnlyTransformerWithLMHead
    (EOTransformerF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    (EOLMHeadF style device dataType inputEmbedDim vocabDim)

type family
  EOTransformerF
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
  EOTransformerF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP = EncoderOnlyTransformer numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP

type family
  EOLMHeadF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOLMHeadF style device dataType inputEmbedDim vocabDim = LMHead style device dataType inputEmbedDim vocabDim

lookupEncoderInputEmbedDim ::
  forall inputEmbedDim m.
  (KnownDim inputEmbedDim, MonadFail m) =>
  m (Dim String Integer)
lookupEncoderInputEmbedDim = case dimVal @inputEmbedDim of
  Dim (Name name) (Size size) -> pure $ Dim name size
  Dim _ _ -> fail "input embedding dimension unspecified"

lookupEncoderOnlyTransformer ::
  forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    KnownDim typeVocabDim,
    Scalar dropoutP,
    HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (EncoderOnlyTransformer numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
lookupEncoderOnlyTransformer dropoutP eps prefix =
  let encoder SBERT = lookupEncoder dropoutP eps prefix
      encoder SRoBERTa = lookupEncoder dropoutP eps prefix
      embedding SBERT = Embedding <$> lookupTensor (prefix <> "embeddings.word_embeddings.weight")
      embedding SRoBERTa = Embedding <$> lookupTensor (prefix <> "embeddings.word_embeddings.weight")
      typeEmbedding SBERT = Embedding <$> lookupTensor (prefix <> "embeddings.token_type_embeddings.weight")
      typeEmbedding SRoBERTa = Embedding <$> lookupTensor (prefix <> "embeddings.token_type_embeddings.weight")
   in EncoderOnlyTransformer
        <$> ( GEncoderOnlyTransformer
                <$> encoder (sing @style)
                <*> embedding (sing @style)
                <*> typeEmbedding (sing @style)
                <*> lookupEncoderInputEmbedDim @inputEmbedDim
            )

lookupEncoderOnlyTransformerWithLMHead ::
  forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    KnownDim typeVocabDim,
    Scalar dropoutP,
    HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (EncoderOnlyTransformerWithLMHead numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
lookupEncoderOnlyTransformerWithLMHead dropoutP eps prefix =
  let transformer SBERT = lookupEncoderOnlyTransformer dropoutP eps (prefix <> "bert.")
      transformer SRoBERTa = lookupEncoderOnlyTransformer dropoutP eps (prefix <> "roberta.")
      lmHead SBERT = lookupLMHead eps (prefix <> "cls.predictions.")
      lmHead SRoBERTa = lookupLMHead eps (prefix <> "lm_head.")
   in EncoderOnlyTransformerWithLMHead
        <$> ( GEncoderOnlyTransformerWithLMHead
                <$> transformer (sing @style)
                <*> lmHead (sing @style)
            )

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
      (EOEmbeddingF style device dataType inputEmbedDim vocabDim)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    embeddingOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    HasForward
      (EOTypeEmbeddingF style device dataType inputEmbedDim typeVocabDim)
      inputType
      embeddingGeneratorOutput
      typeEmbeddingOutput
      typeEmbeddingGeneratorOutput,
    typeEmbeddingOutput ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (EOEncoderF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( Tensor
          (requiresGradient' <|> requiresGradient'')
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
    (EncoderOnlyTransformer numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    (EncoderOnlyTransformerInput input inputType pos attentionMask)
    generator
    (EncoderOnlyTransformerOutput encoderOutput)
    generatorOutput
  where
  forward (EncoderOnlyTransformer GEncoderOnlyTransformer {..}) EncoderOnlyTransformerInput {..} =
    let s :: Double = sqrt . fromIntegral . dimSize $ eoInputEmbedDim
        embedScaling ::
          forall requiresGradient layout device dataType shape.
          STransformerStyle style ->
          Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType shape
        embedScaling SBERT = id
        embedScaling SRoBERTa = id
        -- embedScaling _ = flip mulScalar s
        embeddedInput =
          ireturn eoInput
            >>>= IxState . forward eoEmbedding
            >>>= ireturn . embedScaling (sing @style)
        embeddedInputType =
          ireturn eoInputType
            >>>= IxState . forward eoTypeEmbedding
            >>>= ireturn . embedScaling (sing @style)
     in runIxState $
          add <<$>> embeddedInput <<*>> embeddedInputType
            >>>= (\input' -> IxState $ forward eoEncoder (input', eoPos, eoAttentionMask))
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
      (EOTransformerF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
      input
      generator
      eoOutput
      eoGeneratorOutput,
    eoOutput ~ EncoderOnlyTransformerOutput encoderOutput,
    HasForward
      (EOLMHeadF style device dataType inputEmbedDim vocabDim)
      encoderOutput
      eoGeneratorOutput
      lmHeadOutput
      generatorOutput,
    output ~ EncoderOnlyTransformerOutput lmHeadOutput
  ) =>
  HasForward
    (EncoderOnlyTransformerWithLMHead numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP)
    input
    generator
    output
    generatorOutput
  where
  forward (EncoderOnlyTransformerWithLMHead GEncoderOnlyTransformerWithLMHead {..}) input =
    runIxState $
      ireturn input
        >>>= IxState . forward eoTransformer
        >>>= ( \EncoderOnlyTransformerOutput {..} ->
                 ireturn eoEncoderOutput
                   >>>= IxState . forward eoLMHead
                   >>>= \lmHeadOutput -> ireturn (EncoderOnlyTransformerOutput lmHeadOutput)
             )