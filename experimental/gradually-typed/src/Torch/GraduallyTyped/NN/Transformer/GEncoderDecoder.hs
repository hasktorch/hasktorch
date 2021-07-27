{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
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

module Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat (SNat))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GLMHead (GLMHead, LMHeadActivationF, LMHeadBiasF, LMHeadDecoderF, LMHeadDenseF, LMHeadLayerNormF, lmHeadSpec)
import Torch.GraduallyTyped.NN.Transformer.GTransformer (GTransformer, TDFinalDropoutF, TDFinalLayerNormF, TDInitialDropoutF, TDInitialLayerNormF, TDPosEncF, TDRelPosEncF, TDStackF, TEFinalDropoutF, TEFinalLayerNormF, TEInitialDropoutF, TEInitialLayerNormF, TEPosEncF, TERelPosEncF, TEStackF, transformerDecoderSpec, transformerEncoderSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (..), STransformerStyle (..), TransformerHead (WithLMHead, WithoutHead), TransformerStyle (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (), TensorSpec (..))
import Prelude hiding (head)

-- | Data type that is used to represent whether the encoder-decoder transformer model has a scaled embedding.
data EncoderDecoderTransformerHasEmbedScaling
  = EncoderDecoderTransformerWithEmbedScaling
  | EncoderDecoderTransformerWithoutEmbedScaling

-- | Generic encoder-decoder transformer model.
-- This is a model that can be used to encode and decode sequences of variable length.
--
-- - @inputEmbedDim@: the dimension of the input embedding.
-- - @encoder@: a transformer encoder.
-- - @decoder@: a transformer decoder.
-- - @sharedEmbedding@: a shared embedding layer.
-- - @head@: a head layer for the output.
data
  GEncoderDecoderTransformer
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoder :: Type)
    (decoder :: Type)
    (sharedEmbedding :: Type)
    (head :: Type)
  where
  GEncoderDecoderTransformer ::
    forall inputEmbedDim encoder decoder sharedEmbedding head.
    { -- | input embedding dim for scaling
      edtInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      edtEncoder :: encoder,
      -- | decoder
      edtDecoder :: decoder,
      -- | embedding shared between encoder and decoder
      edtSharedEmbedding :: sharedEmbedding,
      -- | transformer head
      edtHead :: head,
      -- | embedding scaling
      edtEmbedScaling :: EncoderDecoderTransformerHasEmbedScaling
    } ->
    GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head

type instance
  ModelSpec (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head) =
    GEncoderDecoderTransformer inputEmbedDim (ModelSpec encoder) (ModelSpec decoder) (ModelSpec sharedEmbedding) (ModelSpec head)

-- | Specifies the encoder of the encoder-decoder transformer model.
type family
  EDTEncoderF
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
  EDTEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim =
    NamedModel
      ( GTransformer
          (TEPosEncF style gradient device dataType inputEmbedDim posEncDim)
          (TERelPosEncF style gradient device dataType headDim posEncDim)
          (TEInitialLayerNormF style gradient device dataType inputEmbedDim)
          (TEInitialDropoutF style)
          (TEStackF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
          (TEFinalLayerNormF style gradient device dataType inputEmbedDim)
          (TEFinalDropoutF style)
      )

-- | Specifies the decoder of the encoder-decoder transformer model.
type family
  EDTDecoderF
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
  EDTDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim =
    NamedModel
      ( GTransformer
          (TDPosEncF style gradient device dataType inputEmbedDim posEncDim)
          (TDRelPosEncF style gradient device dataType headDim posEncDim)
          (TDInitialLayerNormF style gradient device dataType inputEmbedDim)
          (TDInitialDropoutF style)
          (TDStackF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim)
          (TDFinalLayerNormF style gradient device dataType inputEmbedDim)
          (TDFinalDropoutF style)
      )

-- | Specifies the shared embedding layer of the encoder-decoder transformer model.
type family
  EDTSharedEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  EDTSharedEmbeddingF _ gradient device dataType inputEmbedDim vocabDim =
    NamedModel (Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)

-- | Specifies the head of the encoder-decoder transformer model.
type family
  EDTHeadF
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EDTHeadF style 'WithoutHead gradient device dataType inputEmbedDim vocabDim =
    ()
  EDTHeadF style 'WithLMHead gradient device dataType inputEmbedDim vocabDim =
    NamedModel
      ( GLMHead
          inputEmbedDim
          (LMHeadDenseF style gradient device dataType inputEmbedDim)
          (LMHeadActivationF style)
          (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
          (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
          (LMHeadBiasF style gradient device dataType vocabDim)
      )

-- | Specifies the parameters of an encoder-decoder transformer model.
--
-- - @style@: the style of the encoder-decoder transformer model, e.g. 'ST5', 'SBART', etc.
-- - @transformerHead@: the head of the encoder-decoder transformer model.
-- - @numEncoderLayers@: the number of encoder layers of the encoder-decoder transformer model.
-- - @numDecoderLayers@: the number of decoder layers of the encoder-decoder transformer model.
-- - @gradient@: whether to compute the gradient of the model parameters
-- - @device@: the computational device on which the model is allocated.
-- - @dataType@: the data type of the model parameters.
-- - @headDim@: the dimension of all transformer heads in the encoder-decoder transformer model.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @inputEmbedDim@: the dimension of the input embeddings for both the encoder and the decoder.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @posEncDim@: the dimension of the positional embeddings.
-- - @vocabDim@: the dimension of the vocabulary.
-- - @typeVocabDim@: the dimension of the type vocabulary.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
encoderDecoderTransformerSpec ::
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
  ModelSpec
    ( GEncoderDecoderTransformer
        inputEmbedDim
        (EDTEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
        (EDTDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
        (EDTSharedEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
        (EDTHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)
    )
encoderDecoderTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps =
  let encoderSpec ST5 = NamedModel "encoder." $ encoderSpec' ST5
      encoderSpec SByT5 = NamedModel "encoder." $ encoderSpec' SByT5
      encoderSpec SBART = NamedModel "model.encoder." $ encoderSpec' SBART
      encoderSpec SMBART = NamedModel "model.encoder." $ encoderSpec' SMBART
      encoderSpec SPegasus = NamedModel "model.encoder." $ encoderSpec' SPegasus
      encoderSpec SBERT = undefined
      encoderSpec SRoBERTa = undefined
      encoderSpec SGPT2 = undefined
      decoderSpec ST5 = NamedModel "decoder." $ decoderSpec' ST5
      decoderSpec SByT5 = NamedModel "decoder." $ decoderSpec' SByT5
      decoderSpec SBART = NamedModel "model.decoder." $ decoderSpec' SBART
      decoderSpec SMBART = NamedModel "model.decoder." $ decoderSpec' SMBART
      decoderSpec SPegasus = NamedModel "model.decoder." $ decoderSpec' SPegasus
      decoderSpec SBERT = undefined
      decoderSpec SRoBERTa = undefined
      decoderSpec SGPT2 = undefined
      sharedEmbeddingSpec ST5 = NamedModel "shared." sharedEmbeddingSpec'
      sharedEmbeddingSpec SByT5 = NamedModel "shared." sharedEmbeddingSpec'
      sharedEmbeddingSpec SBART = NamedModel "model.shared." sharedEmbeddingSpec'
      sharedEmbeddingSpec SMBART = NamedModel "model.shared." sharedEmbeddingSpec'
      sharedEmbeddingSpec SPegasus = NamedModel "model.shared." sharedEmbeddingSpec'
      sharedEmbeddingSpec SBERT = undefined
      sharedEmbeddingSpec SRoBERTa = undefined
      sharedEmbeddingSpec SGPT2 = undefined
      headSpec ST5 SWithoutHead = ()
      headSpec ST5 SWithLMHead = NamedModel "lm_head." $ headSpec' ST5
      headSpec SByT5 SWithoutHead = ()
      headSpec SByT5 SWithLMHead = NamedModel "lm_head." $ headSpec' SByT5
      headSpec SBART SWithoutHead = ()
      headSpec SBART SWithLMHead = NamedModel mempty $ headSpec' SBART
      headSpec SMBART SWithoutHead = ()
      headSpec SMBART SWithLMHead = NamedModel mempty $ headSpec' SMBART
      headSpec SPegasus SWithoutHead = ()
      headSpec SPegasus SWithLMHead = NamedModel mempty $ headSpec' SPegasus
      headSpec SBERT _ = undefined
      headSpec SRoBERTa _ = undefined
      headSpec SGPT2 _ = undefined
      embedScalingSpec :: STransformerStyle style -> EncoderDecoderTransformerHasEmbedScaling
      embedScalingSpec ST5 = EncoderDecoderTransformerWithoutEmbedScaling
      embedScalingSpec SByT5 = EncoderDecoderTransformerWithoutEmbedScaling
      embedScalingSpec SBART = EncoderDecoderTransformerWithoutEmbedScaling
      embedScalingSpec SMBART = EncoderDecoderTransformerWithoutEmbedScaling
      embedScalingSpec SPegasus = EncoderDecoderTransformerWithEmbedScaling
      embedScalingSpec SBERT = undefined
      embedScalingSpec SRoBERTa = undefined
      embedScalingSpec SGPT2 = undefined
   in GEncoderDecoderTransformer inputEmbedDim (encoderSpec style) (decoderSpec style) (sharedEmbeddingSpec style) (headSpec style transformerHead) (embedScalingSpec style)
  where
    encoderSpec' :: _
    encoderSpec' style' = transformerEncoderSpec style' numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
    decoderSpec' :: _
    decoderSpec' style' = transformerDecoderSpec style' numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP eps
    sharedEmbeddingSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
    headSpec' :: _
    headSpec' style' = lmHeadSpec style' gradient device dataType inputEmbedDim vocabDim eps

instance
  ( HasInitialize encoder generatorDevice encoder generatorDevice,
    HasInitialize decoder generatorDevice decoder generatorDevice,
    HasInitialize sharedEmbedding generatorDevice sharedEmbedding generatorDevice,
    HasInitialize head generatorDevice head generatorDevice
  ) =>
  HasInitialize
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    generatorDevice
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    generatorDevice
  where
  initialize (GEncoderDecoderTransformer inputEmbedDim encoderSpec decoderSpec sharedEmbeddingSpec headSpec embedScalingSpec) =
    let encoder = IxStateT . initialize $ encoderSpec
        decoder = IxStateT . initialize $ decoderSpec
        sharedEmbedding = IxStateT . initialize $ sharedEmbeddingSpec
        head = IxStateT . initialize $ headSpec
     in runIxStateT
          ( GEncoderDecoderTransformer inputEmbedDim
              <<$>> encoder
              <<*>> decoder
              <<*>> sharedEmbedding
              <<*>> head
              <<*>> ireturn embedScalingSpec
          )

instance
  ( HasStateDict encoder,
    HasStateDict decoder,
    HasStateDict sharedEmbedding,
    HasStateDict head
  ) =>
  HasStateDict (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
  where
  fromStateDict (GEncoderDecoderTransformer inputEmbedDim encoderSpec decoderSpec sharedEmbeddingSpec headSpec embedScalingSpec) k =
    GEncoderDecoderTransformer
      inputEmbedDim
      <$> fromStateDict encoderSpec k
      <*> fromStateDict decoderSpec k
      <*> fromStateDict sharedEmbeddingSpec k
      <*> fromStateDict headSpec k
      <*> pure embedScalingSpec
  toStateDict k GEncoderDecoderTransformer {..} = do
    () <- toStateDict k edtEncoder
    () <- toStateDict k edtDecoder
    () <- toStateDict k edtSharedEmbedding
    () <- toStateDict k edtHead
    pure ()

-- | Input data type for use with an encoder-decoder transformer.
-- Use this for training.
data EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask where
  EncoderDecoderTransformerInput ::
    forall input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask.
    { input :: input,
      decoderInput :: decoderInput,
      pos :: pos,
      decoderPos :: decoderPos,
      attentionMask :: attentionMask,
      decoderAttentionMask :: decoderAttentionMask,
      crossAttentionMask :: crossAttentionMask
    } ->
    EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask

deriving instance
  ( Show input,
    Show decoderInput,
    Show pos,
    Show decoderPos,
    Show attentionMask,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)

-- | Output data type for use with an encoder-decoder transformer.
data EncoderDecoderTransformerOutput decoderOutput encoderOutput where
  EncoderDecoderTransformerOutput ::
    forall decoderOutput encoderOutput.
    { decoderOutput :: decoderOutput,
      encoderOutput :: encoderOutput
    } ->
    EncoderDecoderTransformerOutput decoderOutput encoderOutput

deriving instance
  ( Show decoderOutput,
    Show encoderOutput
  ) =>
  Show (EncoderDecoderTransformerOutput decoderOutput encoderOutput)

-- | Input data type for use with an encoder-decoder transformer.
-- Use this for inference.
data EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask where
  EncoderDecoderTransformerGenerationInput ::
    forall decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask.
    { generationDecoderInput :: decoderInput,
      generationEncoderOutput :: encoderOutput,
      generationDecoderPos :: decoderPos,
      generationDecoderAttentionMask :: decoderAttentionMask,
      generationCrossAttentionMask :: crossAttentionMask
    } ->
    EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show decoderPos,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)

-- | 'HasForward' instance for encoder-decoder transformers with optional head.
--
-- @
--     ┌───────┐  ┌─────┐  ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
--     │ input │  │ pos │  │ attentionMask │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
--     └───┬───┘  └──┬──┘  └──────┬────────┘  └──────┬───────┘  └─────┬──────┘  └──────────┬───────────┘  └─────────┬──────────┘
--         │         │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
-- edtSharedEmbedding│            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--   (embedScaling)  │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--     edtEncoder◄───┘◄───────────┘                  ▼                │                    │                        │
--         │                                 edtSharedEmbedding       │                    │                        │
--         │                                         ▼                │                    │                        │
--         │                                   (embedScaling)         │                    │                        │
--         │                                         ▼                │                    │                        │
--         ├────────────────────────────────────►edtDecoder◄──────────┘◄───────────────────┘◄───────────────────────┘
--         │                                         ▼
--         │                                     (edtHead)
--         │                                         │
--         ▼                                         ▼
-- ┌───────────────┐                         ┌───────────────┐
-- │ encoderOutput │                         │ decoderOutput │
-- └───────────────┘                         └───────────────┘
-- @
instance
  ( HasForward
      sharedEmbedding
      input
      generatorDevice
      embeddingOutput
      embeddingGeneratorOutputDevice,
    embeddingOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    HasForward
      encoder
      (embeddingOutput, pos, attentionMask)
      embeddingGeneratorOutputDevice
      encoderOutput
      encoderGeneratorOutputDevice,
    HasForward
      sharedEmbedding
      decoderInput
      encoderGeneratorOutputDevice
      embeddingOutput'
      embeddingGeneratorOutputDevice',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      decoder
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
      head
      decoderOutput
      decoderGeneratorOutputDevice
      headOutput
      generatorOutputDevice
  ) =>
  HasForward
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    (EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
    generatorDevice
    (EncoderDecoderTransformerOutput headOutput encoderOutput)
    generatorOutputDevice
  where
  forward GEncoderDecoderTransformer {..} EncoderDecoderTransformerInput {..} =
    let scaling :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ edtInputEmbedDim
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward edtSharedEmbedding
            >>>= ireturn
              . ( \case
                    EncoderDecoderTransformerWithoutEmbedScaling -> id
                    EncoderDecoderTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                edtEmbedScaling
            >>>= (\input' -> IxStateT $ forward edtEncoder (input', pos, attentionMask))
            >>>= ( \encoderOutput ->
                     ireturn decoderInput
                       >>>= IxStateT . forward edtSharedEmbedding
                       >>>= ireturn
                         . ( \case
                               EncoderDecoderTransformerWithoutEmbedScaling -> id
                               EncoderDecoderTransformerWithEmbedScaling -> flip mulScalar scaling
                           )
                           edtEmbedScaling
                       >>>= ( \decoderInput' ->
                                IxStateT $ forward edtDecoder (decoderInput', encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask)
                            )
                       >>>= IxStateT . forward edtHead
                       >>>= \decoderOutput -> ireturn (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
                 )

-- | 'HasForward' instance for encoder-decoder transformers with optional head.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- @
-- ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ encoderOutput │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └───────┬───────┘  └───────┬──────┘  └──────┬─────┘  └───────────┬──────────┘  └──────────┬─────────┘
--         │                  │                │                    │                        │
--         │                  ▼                │                    │                        │
--         │          edtSharedEmbedding        │                    │                        │
--         │                  ▼                │                    │                        │
--         │            (embedScaling)         │                    │                        │
--         │                  ▼                │                    │                        │
--         ├──────────►edtDecoder◄────────┘◄───────────────────┘◄───────────────────────┘
--         │                  │
--         │            (edtHead)
--         │                  │
--         ▼                  ▼
-- ┌───────────────┐  ┌───────────────┐
-- │ encoderOutput │  │ decoderOutput │
-- └───────────────┘  └───────────────┘
-- @
instance
  ( HasForward
      sharedEmbedding
      decoderInput
      generatorDevice
      embeddingOutput'
      embeddingGeneratorOutputDevice',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      decoder
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
      head
      decoderOutput
      decoderGeneratorOutputDevice
      headOutput
      generatorOutputDevice
  ) =>
  HasForward
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    (EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
    generatorDevice
    (EncoderDecoderTransformerOutput headOutput encoderOutput)
    generatorOutputDevice
  where
  forward GEncoderDecoderTransformer {..} EncoderDecoderTransformerGenerationInput {..} =
    let scaling :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ edtInputEmbedDim
     in runIxStateT $
          ireturn generationDecoderInput
            >>>= IxStateT . forward edtSharedEmbedding
            >>>= ireturn
              . ( \case
                    EncoderDecoderTransformerWithoutEmbedScaling -> id
                    EncoderDecoderTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                edtEmbedScaling
            >>>= ( \decoderInput' ->
                     IxStateT $ forward edtDecoder (decoderInput', generationEncoderOutput, generationDecoderPos, generationDecoderAttentionMask, generationCrossAttentionMask)
                 )
            >>>= IxStateT . forward edtHead
            >>>= \decoderOutput -> ireturn (EncoderDecoderTransformerOutput decoderOutput generationEncoderOutput)

testEncoderDecoderTransformer :: IO _
testEncoderDecoderTransformer = do
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
      dropoutP = 0
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
    let spec = NamedModel "t5." $ encoderDecoderTransformerSpec ST5 SWithLMHead (SNat @32) (SNat @32) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps
    (t5, g') <- initialize spec g
    let pos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
    forward t5 EncoderDecoderTransformerInput {..} g'
  (bartOutput, g'''') <- do
    let spec = NamedModel "bart." $ encoderDecoderTransformerSpec SBART SWithLMHead (SNat @32) (SNat @32) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps
    (bart, g''') <- initialize spec g''
    let pos = sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
    forward bart EncoderDecoderTransformerInput {..} g'''
  pure ((t5Output, bartOutput), g'''')
