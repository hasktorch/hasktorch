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
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, TransformerEncoderSpec (..))
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead, LMHeadSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Stack ()
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (..), STransformerStyle (..), TransformerHead (..), TransformerStyle (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sGeneratorToDevice)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Prelude hiding (head)

-- | Generic encoder-only transformer model.
data
  GEncoderOnlyTransformer
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoder :: Type)
    (encoderEmbedding :: Type)
    (encoderTypeEmbedding :: Type)
    (head :: Type)
  where
  GEncoderOnlyTransformer ::
    forall inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head.
    { -- | input embedding dim for scaling
      eoInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      eoEncoder :: encoder,
      -- | encoder embedding
      eoEmbedding :: encoderEmbedding,
      -- | encoder type embedding
      eoTypeEmbedding :: encoderTypeEmbedding,
      -- | encoder head
      eoHead :: head
    } ->
    GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head

-- | Encoder-only transformer model.
data
  EncoderOnlyTransformer
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
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
    forall style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    GEncoderOnlyTransformer
      inputEmbedDim
      (EOEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
      (EOEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (EOTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim)
      (EOHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim) ->
    EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

data
  EncoderOnlyTransformerSpec
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
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
    forall style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim.
    STransformerStyle style ->
    STransformerHead transformerHead ->
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
    EncoderOnlyTransformerSpec style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

type instance ModelSpec (EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim) = EncoderOnlyTransformerSpec style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim

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

type family
  EOHeadF
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOHeadF style 'WithoutHead gradient device dataType inputEmbedDim vocabDim =
    ()
  EOHeadF style 'WithLMHead gradient device dataType inputEmbedDim vocabDim =
    LMHead style gradient device dataType inputEmbedDim vocabDim

instance
  ( encoder ~ TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim,
    HasInitialize encoder device encoder device,
    embedding ~ Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
    HasInitialize embedding device embedding device,
    typeEmbedding ~ Embedding gradient ('Layout 'Dense) device dataType typeVocabDim inputEmbedDim 'Nothing,
    HasInitialize embedding device embedding device,
    head ~ EOHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim,
    HasInitialize head device head device
  ) =>
  HasInitialize
    (EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
    generatorDevice
    (EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
    device
  where
  initialize (EncoderOnlyTransformerSpec style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        encoder = IxStateT . initialize @encoder $ TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        embedding = IxStateT . initialize @embedding $ EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
        typeEmbedding = IxStateT . initialize @typeEmbedding $ EmbeddingSpec gradient (SLayout SDense) device dataType typeVocabDim inputEmbedDim SNothing
        head = IxStateT . initialize @head $ case transformerHead of
          SWithoutHead -> ()
          SWithLMHead -> LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps
        geot = GEncoderOnlyTransformer <<$>> ireturn inputEmbedDim <<*>> encoder <<*>> embedding <<*>> typeEmbedding <<*>> head
     in runIxStateT (geot >>>= ireturn . EncoderOnlyTransformer) generator'

instance
  (SingI style, SingI transformerHead) =>
  HasStateDict
    (EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
  where
  fromStateDict (EncoderOnlyTransformerSpec style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim dropoutP eps) k =
    let encoderSpec = TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
        encoder ST5 = undefined
        encoder SByT5 = undefined
        encoder SBART = undefined
        encoder SMBART = undefined
        encoder SPegasus = undefined
        encoder SBERT = fromStateDict encoderSpec (k <> "bert.")
        encoder SRoBERTa = fromStateDict encoderSpec (k <> "roberta.")
        encoder SGPT2 = undefined
        embeddingSpec = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
        embedding ST5 = undefined
        embedding SByT5 = undefined
        embedding SBART = undefined
        embedding SMBART = undefined
        embedding SPegasus = undefined
        embedding SBERT = fromStateDict embeddingSpec (k <> "bert.embeddings.word_embeddings.")
        embedding SRoBERTa = fromStateDict embeddingSpec (k <> "roberta.embeddings.word_embeddings.")
        embedding SGPT2 = undefined
        typeEmbeddingSpec = EmbeddingSpec gradient (SLayout SDense) device dataType typeVocabDim inputEmbedDim SNothing
        typeEmbedding ST5 = undefined
        typeEmbedding SByT5 = undefined
        typeEmbedding SBART = undefined
        typeEmbedding SMBART = undefined
        typeEmbedding SPegasus = undefined
        typeEmbedding SBERT = fromStateDict typeEmbeddingSpec (k <> "bert.embeddings.token_type_embeddings.")
        typeEmbedding SRoBERTa = fromStateDict typeEmbeddingSpec (k <> "roberta.embeddings.token_type_embeddings.")
        typeEmbedding SGPT2 = undefined
        lmHeadSpec = LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps
        head _ SWithoutHead = fromStateDict () k
        head ST5 _ = undefined
        head SByT5 _ = undefined
        head SBART _ = undefined
        head SMBART _ = undefined
        head SPegasus _ = undefined
        head SBERT SWithLMHead = fromStateDict lmHeadSpec (k <> "cls.predictions.")
        head SRoBERTa SWithLMHead = fromStateDict lmHeadSpec (k <> "lm_head.")
        head SGPT2 _ = undefined
     in EncoderOnlyTransformer
          <$> ( GEncoderOnlyTransformer
                  inputEmbedDim
                  <$> encoder (sing @style)
                  <*> embedding (sing @style)
                  <*> typeEmbedding (sing @style)
                  <*> head style transformerHead
              )
  toStateDict k (EncoderOnlyTransformer GEncoderOnlyTransformer {..}) =
    let encoder ST5 = undefined
        encoder SByT5 = undefined
        encoder SBART = undefined
        encoder SMBART = undefined
        encoder SPegasus = undefined
        encoder SBERT = toStateDict (k <> "bert.")
        encoder SRoBERTa = toStateDict (k <> "roberta.")
        encoder SGPT2 = undefined
        embedding ST5 = undefined
        embedding SByT5 = undefined
        embedding SBART = undefined
        embedding SMBART = undefined
        embedding SPegasus = undefined
        embedding SBERT = toStateDict (k <> "bert.embeddings.word_embeddings.")
        embedding SRoBERTa = toStateDict (k <> "roberta.embeddings.word_embeddings.")
        embedding SGPT2 = undefined
        typeEmbedding ST5 = undefined
        typeEmbedding SByT5 = undefined
        typeEmbedding SBART = undefined
        typeEmbedding SMBART = undefined
        typeEmbedding SPegasus = undefined
        typeEmbedding SBERT = toStateDict (k <> "bert.embeddings.token_type_embeddings.")
        typeEmbedding SRoBERTa = toStateDict (k <> "roberta.embeddings.token_type_embeddings.")
        typeEmbedding SGPT2 = undefined
        head _ SWithoutHead = toStateDict k
        head ST5 _ = undefined
        head SByT5 _ = undefined
        head SBART _ = undefined
        head SMBART _ = undefined
        head SPegasus _ = undefined
        head SBERT SWithLMHead = toStateDict (k <> "cls.predictions.")
        head SRoBERTa SWithLMHead = toStateDict (k <> "lm_head.")
        head SGPT2 _ = undefined
     in do
          () <- encoder (sing @style) eoEncoder
          () <- embedding (sing @style) eoEmbedding
          () <- typeEmbedding (sing @style) eoTypeEmbedding
          () <- head (sing @style) (sing @transformerHead) eoHead
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

-- | 'HasForward' instance for encoder-only transformers with optional head.
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
--         ▼
--      (eoHead)
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
      eoGeneratorOutputDevice,
    HasForward
      (EOHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)
      encoderOutput
      eoGeneratorOutputDevice
      headOutput
      generatorOutputDevice
  ) =>
  HasForward
    (EncoderOnlyTransformer style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim)
    (EncoderOnlyTransformerInput input inputType pos attentionMask)
    generatorDevice
    (EncoderOnlyTransformerOutput headOutput)
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
            >>>= IxStateT . forward eoHead
            >>>= ireturn . EncoderOnlyTransformerOutput
