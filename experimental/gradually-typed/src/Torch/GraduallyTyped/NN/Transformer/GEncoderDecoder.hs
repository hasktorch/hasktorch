{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
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
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingKind (fromSing))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.Index.Type (Index (NegativeIndex), SIndex (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, logSoftmax)
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GLMHead (GLMHeadF, lmHeadSpec)
import Torch.GraduallyTyped.NN.Transformer.GTransformer (TransformerDecoderF, TransformerEncoderF, transformerDecoderSpec, transformerEncoderSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (..), STransformerStyle (..), ShiftRight, TransformerHead (WithLMHead, WithoutHead), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked, pattern (:|:))
import Torch.GraduallyTyped.Prelude.List (SList (SNil))
import Torch.GraduallyTyped.Prelude.Maybe (SMaybe (SNothing))
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim, SSelectDim (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Indexing (IndexDims, IndexType (..), Indices (..), SIndexType (..), SIndices (..), (!))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (GatherDimF, SqueezeDimF, UnsqueezeF, sGatherDim, sSqueezeDim, sUnsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar)
import Torch.GraduallyTyped.Tensor.MathOperations.Reduction (MeanAllCheckF, meanAll)
import Torch.GraduallyTyped.Tensor.Type (Tensor ())
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Prelude hiding (head)

-- | Data type that is used to represent whether the encoder-decoder transformer model has a scaled embedding.
data EncoderDecoderTransformerHasEmbedScaling
  = EncoderDecoderTransformerWithEmbedScaling
  | EncoderDecoderTransformerWithoutEmbedScaling
  deriving stock (Eq, Ord, Show, Generic)

type instance ModelSpec EncoderDecoderTransformerHasEmbedScaling = EncoderDecoderTransformerHasEmbedScaling

instance HasInitialize EncoderDecoderTransformerHasEmbedScaling generatorDevice EncoderDecoderTransformerHasEmbedScaling generatorDevice where
  initialize hasEmbedScaling g = pure (hasEmbedScaling, g)

instance HasStateDict EncoderDecoderTransformerHasEmbedScaling where
  fromStateDict hasEmbedScaling _ = pure hasEmbedScaling
  toStateDict _ _ = pure ()

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
  deriving stock (Show, Generic)

type instance
  ModelSpec (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head) =
    GEncoderDecoderTransformer inputEmbedDim (ModelSpec encoder) (ModelSpec decoder) (ModelSpec sharedEmbedding) (ModelSpec head)

type family
  GEncoderDecoderTransformerF
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
    (hasDropout :: HasDropout) ::
    Type
  where
  GEncoderDecoderTransformerF style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim hasDropout =
    GEncoderDecoderTransformer
      inputEmbedDim
      (EDTEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout)
      (EDTDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout)
      (EDTSharedEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (EDTHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)

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
    (hasDropout :: HasDropout) ::
    Type
  where
  EDTEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout =
    NamedModel (TransformerEncoderF style numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout)

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
    (hasDropout :: HasDropout) ::
    Type
  where
  EDTDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout =
    NamedModel (TransformerDecoderF style numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim hasDropout)

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
    NamedModel (GLMHeadF style gradient device dataType inputEmbedDim vocabDim)

-- | Specifies the parameters of an encoder-decoder transformer model.
encoderDecoderTransformerSpec ::
  forall style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim hasDropout.
  -- | the style of the encoder-decoder transformer model, e.g. 'ST5', 'SBART', etc.
  STransformerStyle style ->
  -- | the head of the encoder-decoder transformer model.
  STransformerHead transformerHead ->
  -- | the number of encoder layers of the encoder-decoder transformer model.
  SNat numEncoderLayers ->
  -- | the number of decoder layers of the encoder-decoder transformer model.
  SNat numDecoderLayers ->
  -- | whether or not to compute the gradient of the model parameters
  SGradient gradient ->
  -- | the computational device on which the model is allocated.
  SDevice device ->
  -- | the data type of the model parameters.
  SDataType dataType ->
  -- | the dimension of all transformer heads in the encoder-decoder transformer model.
  SDim headDim ->
  -- | the dimension of the transformer head embeddings.
  SDim headEmbedDim ->
  -- | the dimension of the transformer embeddings.
  SDim embedDim ->
  -- | the dimension of the input embeddings for both the encoder and the decoder.
  SDim inputEmbedDim ->
  -- | the dimension of the feed-forward network.
  SDim ffnDim ->
  -- | the dimension of the positional embeddings.
  SDim posEncDim ->
  -- | the dimension of the vocabulary.
  SDim vocabDim ->
  -- | whether or not to use dropout.
  SHasDropout hasDropout ->
  -- | the dropout rate.
  Double ->
  -- | the epsilon value for numerical stability of the layer normalization.
  Double ->
  -- | the parameter specification of an encoder-decoder transformer model.
  ModelSpec (GEncoderDecoderTransformerF style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim hasDropout)
encoderDecoderTransformerSpec style transformerHead numEncoderLayers numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim hasDropout dropoutP eps =
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
    encoderSpec' style' = transformerEncoderSpec style' numEncoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout dropoutP eps
    decoderSpec' :: _
    decoderSpec' style' = transformerDecoderSpec style' numDecoderLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim hasDropout dropoutP eps
    sharedEmbeddingSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
    headSpec' :: _
    headSpec' style' = lmHeadSpec style' gradient device dataType inputEmbedDim vocabDim eps

instance
  ( HasInitialize encoder generatorDevice encoder' generatorDevice0,
    HasInitialize decoder generatorDevice0 decoder' generatorDevice1,
    HasInitialize sharedEmbedding generatorDevice1 sharedEmbedding' generatorDevice2,
    HasInitialize head generatorDevice2 head' generatorOutputDevice
  ) =>
  HasInitialize
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    generatorDevice
    (GEncoderDecoderTransformer inputEmbedDim encoder' decoder' sharedEmbedding' head')
    generatorOutputDevice

instance
  ( HasStateDict encoder,
    HasStateDict decoder,
    HasStateDict sharedEmbedding,
    HasStateDict head
  ) =>
  HasStateDict (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)

data
  GSimplifiedEncoderDecoderTransformer
    (model :: Type)
    (mkPos :: Type)
    (mkDecoderPos :: Type)
    (mkPaddingMask :: Type)
    (mkAttentionMask :: Type)
    (mkCrossAttentionMask :: Type)
    (mkDecoderAttentionMask :: Type)
  where
  GSimplifiedEncoderDecoderTransformer ::
    forall model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask.
    { -- | encoder-decoder model
      sedtModel :: model,
      -- | shift for decoder input
      sedtDecoderInputShift :: ShiftRight Int,
      -- | shift for padding mask
      sedtPaddingMaskShift :: ShiftRight Int,
      -- | make encoder input positions
      sedtMkPos :: mkPos,
      -- | make decoder input position
      sedtMkDecoderPos :: mkDecoderPos,
      -- | make padding mask
      sedtMkPaddingMask :: mkPaddingMask,
      -- | make attention mask
      sedtMkAttentionMask :: mkAttentionMask,
      -- | make cross-attention mask
      sedtMkCrossAttentionMask :: mkCrossAttentionMask,
      -- | make decoder attention mask
      sedtMkDecoderAttentionMask :: mkDecoderAttentionMask
    } ->
    GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask) =
    GSimplifiedEncoderDecoderTransformer (ModelSpec model) (ModelSpec mkPos) (ModelSpec mkDecoderPos) (ModelSpec mkPaddingMask) (ModelSpec mkAttentionMask) (ModelSpec mkCrossAttentionMask) (ModelSpec mkDecoderAttentionMask)

instance
  ( HasInitialize model generatorDevice model' generatorDevice0,
    HasInitialize mkPos generatorDevice0 mkPos' generatorDevice1,
    HasInitialize mkDecoderPos generatorDevice1 mkDecoderPos' generatorDevice2,
    HasInitialize mkPaddingMask generatorDevice2 mkPaddingMask' generatorDevice3,
    HasInitialize mkAttentionMask generatorDevice3 mkAttentionMask' generatorDevice4,
    HasInitialize mkCrossAttentionMask generatorDevice4 mkCrossAttentionMask' generatorDevice5,
    HasInitialize mkDecoderAttentionMask generatorDevice5 mkDecoderAttentionMask' generatorOutputDevice
  ) =>
  HasInitialize
    (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
    generatorDevice
    (GSimplifiedEncoderDecoderTransformer model' mkPos' mkDecoderPos' mkPaddingMask' mkAttentionMask' mkCrossAttentionMask' mkDecoderAttentionMask')
    generatorOutputDevice

instance
  ( HasStateDict model,
    HasStateDict mkPos,
    HasStateDict mkDecoderPos,
    HasStateDict mkPaddingMask,
    HasStateDict mkAttentionMask,
    HasStateDict mkCrossAttentionMask,
    HasStateDict mkDecoderAttentionMask
  ) =>
  HasStateDict (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)

-- | Input data type for use with an encoder-decoder transformer.
-- Use this for training.
data EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask where
  EncoderDecoderTransformerInput ::
    forall input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask.
    { edtInput :: input,
      edtDecoderInput :: decoderInput,
      edtPos :: pos,
      edtDecoderPos :: decoderPos,
      edtAttentionMask :: attentionMask,
      edtDecoderAttentionMask :: decoderAttentionMask,
      edtCrossAttentionMask :: crossAttentionMask
    } ->
    EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask
  deriving stock (Eq, Ord, Show, Generic)

data EncoderDecoderTransformerInput' input pos attentionMask where
  EncoderDecoderTransformerInput' ::
    forall input pos attentionMask.
    { edtInput' :: input,
      edtPos' :: pos,
      edtAttentionMask' :: attentionMask
    } ->
    EncoderDecoderTransformerInput' input pos attentionMask
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerInput input decoderInput where
  SimplifiedEncoderDecoderTransformerInput ::
    forall input decoderInput.
    { sedtInput :: input,
      sedtDecoderInput :: decoderInput
    } ->
    SimplifiedEncoderDecoderTransformerInput input decoderInput
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerInput' input where
  SimplifiedEncoderDecoderTransformerInput' ::
    forall input.
    { sedtInput' :: input
    } ->
    SimplifiedEncoderDecoderTransformerInput' input
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerTrainingInput input target where
  SimplifiedEncoderDecoderTransformerTrainingInput ::
    forall input target.
    { sedtTrainingInput :: input,
      sedtTarget :: target
    } ->
    SimplifiedEncoderDecoderTransformerTrainingInput input target
  deriving stock (Eq, Ord, Show, Generic)

-- | Output data type for use with an encoder-decoder transformer.
data EncoderDecoderTransformerOutput decoderOutput encoderOutput where
  EncoderDecoderTransformerOutput ::
    forall decoderOutput encoderOutput.
    { edtDecoderOutput :: decoderOutput,
      edtEncoderOutput :: encoderOutput
    } ->
    EncoderDecoderTransformerOutput decoderOutput encoderOutput
  deriving stock (Eq, Ord, Show, Generic)

data EncoderDecoderTransformerOutput' encoderOutput where
  EncoderDecoderTransformerOutput' ::
    forall encoderOutput.
    { edtEncoderOutput' :: encoderOutput
    } ->
    EncoderDecoderTransformerOutput' encoderOutput
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask where
  SimplifiedEncoderDecoderTransformerOutput ::
    forall decoderOutput encoderOutput decoderInput inputPaddingMask.
    { sedtDecoderOutput :: decoderOutput,
      sedtEncoderOutput :: encoderOutput,
      sedtOriginalDecoderInput :: decoderInput,
      sedtInputPaddingMask :: inputPaddingMask
    } ->
    SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerOutput' encoderOutput inputPaddingMask where
  SimplifiedEncoderDecoderTransformerOutput' ::
    forall encoderOutput inputPaddingMask.
    { sedtEncoderOutput' :: encoderOutput,
      sedtInputPaddingMask' :: inputPaddingMask
    } ->
    SimplifiedEncoderDecoderTransformerOutput' encoderOutput inputPaddingMask
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerTrainingOutput loss where
  SimplifiedEncoderDecoderTransformerTrainingOutput ::
    forall loss.
    { sedtLoss :: loss
    } ->
    SimplifiedEncoderDecoderTransformerTrainingOutput loss
  deriving stock (Eq, Ord, Show, Generic)

-- | Input data type for use with an encoder-decoder transformer.
-- Use this for inference.
data EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask where
  EncoderDecoderTransformerGenerationInput ::
    forall decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask.
    { edtGenerationDecoderInput :: decoderInput,
      edtGenerationEncoderOutput :: encoderOutput,
      edtGenerationDecoderPos :: decoderPos,
      edtGenerationDecoderAttentionMask :: decoderAttentionMask,
      edtGenerationCrossAttentionMask :: crossAttentionMask
    } ->
    EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask where
  SimplifiedEncoderDecoderTransformerGenerationInput ::
    forall decoderInput encoderOutput inputPaddingMask.
    { sedtGenerationDecoderInput :: decoderInput,
      sedtGenerationEncoderOutput :: encoderOutput,
      sedtGenerationInputPaddingMask :: inputPaddingMask
    } ->
    SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask
  deriving stock (Eq, Ord, Show, Generic)

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
      (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
      (EncoderDecoderTransformerInput' input pos attentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput' encoderOutput)
      generatorDevice',
    HasForward
      (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
      (EncoderDecoderTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generatorDevice'
      (EncoderDecoderTransformerOutput headOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    (EncoderDecoderTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
    generatorDevice
    (EncoderDecoderTransformerOutput headOutput encoderOutput)
    generatorOutputDevice
  where
  forward model EncoderDecoderTransformerInput {..} =
    runIxStateT $
      IxStateT
        ( forward
            model
            EncoderDecoderTransformerInput'
              { edtInput' = edtInput,
                edtPos' = edtPos,
                edtAttentionMask' = edtAttentionMask
              }
        )
        >>>= ( \EncoderDecoderTransformerOutput' {..} ->
                 IxStateT
                   ( forward
                       model
                       EncoderDecoderTransformerGenerationInput
                         { edtGenerationDecoderInput = edtDecoderInput,
                           edtGenerationEncoderOutput = edtEncoderOutput',
                           edtGenerationDecoderPos = edtDecoderPos,
                           edtGenerationDecoderAttentionMask = edtDecoderAttentionMask,
                           edtGenerationCrossAttentionMask = edtCrossAttentionMask
                         }
                   )
             )

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
      generatorOutputDevice
  ) =>
  HasForward
    (GEncoderDecoderTransformer inputEmbedDim encoder decoder sharedEmbedding head)
    (EncoderDecoderTransformerInput' input pos attentionMask)
    generatorDevice
    (EncoderDecoderTransformerOutput' encoderOutput)
    generatorOutputDevice
  where
  forward GEncoderDecoderTransformer {..} EncoderDecoderTransformerInput' {..} =
    let scaling :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ edtInputEmbedDim
     in runIxStateT $
          ireturn edtInput'
            >>>= IxStateT . forward edtSharedEmbedding
            >>>= ilift
              . ( \case
                    EncoderDecoderTransformerWithoutEmbedScaling -> pure
                    EncoderDecoderTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                edtEmbedScaling
            >>>= (\input' -> IxStateT $ forward edtEncoder (input', edtPos', edtAttentionMask'))
            >>>= (\encoderOutput -> ireturn (EncoderDecoderTransformerOutput' encoderOutput))

-- | 'HasForward' instance for encoder-decoder transformers with optional head.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- @
-- ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ encoderOutput │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └───────┬───────┘  └───────┬──────┘  └──────┬─────┘  └───────────┬──────────┘  └──────────┬─────────┘
--         │                  │                │                    │                        │
--         │                  ▼                │                    │                        │
--         │          edtSharedEmbedding       │                    │                        │
--         │                  ▼                │                    │                        │
--         │            (embedScaling)         │                    │                        │
--         │                  ▼                │                    │                        │
--         ├────────────►edtDecoder◄───────────┘◄───────────────────┘◄───────────────────────┘
--         │                  │
--         │              (edtHead)
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
          ireturn edtGenerationDecoderInput
            >>>= IxStateT . forward edtSharedEmbedding
            >>>= ilift
              . ( \case
                    EncoderDecoderTransformerWithoutEmbedScaling -> pure
                    EncoderDecoderTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                edtEmbedScaling
            >>>= ( \decoderInput' ->
                     IxStateT $ forward edtDecoder (decoderInput', edtGenerationEncoderOutput, edtGenerationDecoderPos, edtGenerationDecoderAttentionMask, edtGenerationCrossAttentionMask)
                 )
            >>>= IxStateT . forward edtHead
            >>>= \decoderOutput -> ireturn (EncoderDecoderTransformerOutput decoderOutput edtGenerationEncoderOutput)

-- | 'HasForward' instance for simplified encoder-decoder models.

-- This instance shifts decoder inputs by one token to the right by adding
-- a model-specific sequence initialization token at the beginning.
instance
  ( HasForward
      (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
      (SimplifiedEncoderDecoderTransformerInput' input)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput' encoderOutput inputPaddingMask)
      generatorDevice',
    HasForward
      (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
      (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask)
      generatorDevice'
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
      generatorOutputDevice
  ) =>
  HasForward
    (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
    (SimplifiedEncoderDecoderTransformerInput input decoderInput)
    generatorDevice
    (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
    generatorOutputDevice
  where
  forward model SimplifiedEncoderDecoderTransformerInput {..} =
    runIxStateT $
      IxStateT (forward model SimplifiedEncoderDecoderTransformerInput' {sedtInput' = sedtInput})
        >>>= ( \SimplifiedEncoderDecoderTransformerOutput' {..} ->
                 ireturn $
                   SimplifiedEncoderDecoderTransformerGenerationInput
                     { sedtGenerationDecoderInput = sedtDecoderInput,
                       sedtGenerationEncoderOutput = sedtEncoderOutput',
                       sedtGenerationInputPaddingMask = sedtInputPaddingMask'
                     }
             )
        >>>= IxStateT . forward model

instance
  ( HasForward
      mkPaddingMask
      input
      generatorDevice
      inputPaddingMask
      generatorDevice,
    HasForward
      mkAttentionMask
      inputPaddingMask
      generatorDevice
      attentionMask
      generatorDevice,
    HasForward
      mkPos
      input
      generatorDevice
      pos
      generatorDevice,
    HasForward
      model
      (EncoderDecoderTransformerInput' input pos attentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput' encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
    (SimplifiedEncoderDecoderTransformerInput' input)
    generatorDevice
    (SimplifiedEncoderDecoderTransformerOutput' encoderOutput inputPaddingMask)
    generatorOutputDevice
  where
  forward GSimplifiedEncoderDecoderTransformer {..} SimplifiedEncoderDecoderTransformerInput' {..} =
    runIxStateT $
      IxStateT (forward sedtMkPaddingMask sedtInput')
        >>>= ( \inputPaddingMask ->
                 let pos = IxStateT . forward sedtMkPos $ sedtInput'
                     attentionMask = IxStateT . forward sedtMkAttentionMask $ inputPaddingMask
                  in EncoderDecoderTransformerInput'
                       sedtInput'
                       <<$>> pos
                       <<*>> attentionMask
                       >>>= IxStateT . forward sedtModel
                       >>>= ( \EncoderDecoderTransformerOutput' {..} ->
                                ireturn $
                                  SimplifiedEncoderDecoderTransformerOutput'
                                    { sedtEncoderOutput' = edtEncoderOutput',
                                      sedtInputPaddingMask' = inputPaddingMask
                                    }
                            )
             )

-- | 'HasForward' instance for simplified encoder-decoder models.
-- Use this instance for sequence generation once the encoder's output is available.

-- This instance shifts decoder inputs by one token to the right by adding
-- a model-specific sequence initialization token at the beginning.
instance
  ( HasForward
      mkPaddingMask
      decoderInput
      generatorDevice
      decoderInputPaddingMask
      generatorDevice,
    HasForward
      mkCrossAttentionMask
      (rightShiftedDecoderInput, inputPaddingMask)
      generatorDevice
      crossAttentionMask
      generatorDevice,
    HasForward
      mkDecoderAttentionMask
      rightShiftedDecoderInputPaddingMask
      generatorDevice
      decoderAttentionMask
      generatorDevice,
    HasForward
      (ShiftRight Int)
      decoderInput
      generatorDevice
      rightShiftedDecoderInput
      generatorDevice,
    HasForward
      (ShiftRight Int)
      decoderInputPaddingMask
      generatorDevice
      rightShiftedDecoderInputPaddingMask
      generatorDevice,
    HasForward
      mkDecoderPos
      rightShiftedDecoderInput
      generatorDevice
      decoderPos
      generatorDevice,
    HasForward
      model
      (EncoderDecoderTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
    (SimplifiedEncoderDecoderTransformerGenerationInput decoderInput encoderOutput inputPaddingMask)
    generatorDevice
    (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
    generatorOutputDevice
  where
  forward GSimplifiedEncoderDecoderTransformer {..} SimplifiedEncoderDecoderTransformerGenerationInput {..} =
    runIxStateT $
      ( let rightShiftedDecoderInput = IxStateT . forward sedtDecoderInputShift $ sedtGenerationDecoderInput
            rightShiftedDecoderInputPaddingMask =
              ireturn sedtGenerationDecoderInput
                >>>= IxStateT . forward sedtMkPaddingMask
                >>>= IxStateT . forward sedtPaddingMaskShift
         in (,)
              <<$>> rightShiftedDecoderInput
              <<*>> rightShiftedDecoderInputPaddingMask
      )
        >>>= ( \(rightShiftedDecoderInput, rightShiftedDecoderInputPaddingMask) ->
                 let decoderPos = IxStateT . forward sedtMkDecoderPos $ rightShiftedDecoderInput
                     crossAttentionMask = IxStateT . forward sedtMkCrossAttentionMask $ (rightShiftedDecoderInput, sedtGenerationInputPaddingMask)
                     decoderAttentionMask = IxStateT . forward sedtMkDecoderAttentionMask $ rightShiftedDecoderInputPaddingMask
                  in ( EncoderDecoderTransformerGenerationInput
                         rightShiftedDecoderInput
                         sedtGenerationEncoderOutput
                         <<$>> decoderPos
                         <<*>> decoderAttentionMask
                         <<*>> crossAttentionMask
                     )
                       >>>= IxStateT . forward sedtModel
                       >>>= ( \(EncoderDecoderTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput sedtGenerationDecoderInput sedtGenerationInputPaddingMask
                            )
             )

instance
  ( HasForward
      (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
      (SimplifiedEncoderDecoderTransformerInput input decoderInput)
      generatorDevice
      (SimplifiedEncoderDecoderTransformerOutput decoderOutput encoderOutput decoderInput inputPaddingMask)
      generatorOutputDevice,
    decoderInput
      ~ Tensor
          targetGradient
          targetLayout
          targetDevice
          targetDataType
          (IndexDims ('Indices '[ 'SliceAll, 'SliceUpTo ('NegativeIndex 1)]) targetShape),
    decoderOutput
      ~ Tensor
          doGradient
          doLayout
          doDevice
          doDataType
          doShape,
    logProbsShape ~ SoftmaxF ('SelectDim ('ByIndex 2)) doShape,
    Catch logProbsShape,
    unsqueezedTargetShape ~ UnsqueezeF ('SelectDim ('ByIndex 2)) targetShape,
    Catch unsqueezedTargetShape,
    gatheredLogProbsShape ~ GatherDimF ('SelectDim ('ByIndex 2)) unsqueezedTargetShape logProbsShape,
    Catch gatheredLogProbsShape,
    Catch (targetDataType <+> 'DataType 'Int64),
    logLikelihoodShape ~ SqueezeDimF ('SelectDim ('ByIndex 2)) gatheredLogProbsShape,
    Catch logLikelihoodShape,
    MeanAllCheckF logLikelihoodShape,
    loss
      ~ Tensor
          (targetGradient <|> doGradient)
          (targetLayout <+> doLayout)
          (targetDevice <+> doDevice)
          doDataType
          ('Shape '[]),
    generatorOutputDevice ~ generatorDevice
  ) =>
  HasForward
    (GSimplifiedEncoderDecoderTransformer model mkPos mkDecoderPos mkPaddingMask mkAttentionMask mkCrossAttentionMask mkDecoderAttentionMask)
    ( SimplifiedEncoderDecoderTransformerTrainingInput
        input
        (Tensor targetGradient targetLayout targetDevice targetDataType targetShape)
    )
    generatorDevice
    (SimplifiedEncoderDecoderTransformerTrainingOutput loss)
    generatorOutputDevice
  where
  forward eot SimplifiedEncoderDecoderTransformerTrainingInput {..} =
    runIxStateT $
      ireturn sedtTarget
        >>>= ilift . (! SIndices (SSliceAll :|: SSliceUpTo (SNegativeIndex @1) :|: SNil))
        >>>= (\sedtDecoderInput -> IxStateT . forward eot $ SimplifiedEncoderDecoderTransformerInput {sedtInput = sedtTrainingInput, sedtDecoderInput})
        >>>= ireturn . sedtDecoderOutput
        >>>= ilift
          . ( \logits -> do
                logProbs <- logSoftmax (SSelectDim $ SByIndex @2) logits
                target' <- sUnsqueeze (SSelectDim $ SByIndex @2) sedtTarget
                gatheredLogProbs <- sGatherDim (SSelectDim $ SByIndex @2) target' logProbs
                logLikelihood <- sSqueezeDim (SSelectDim $ SByIndex @2) gatheredLogProbs
                pure . negate $ meanAll logLikelihood
            )
        >>>= ireturn . SimplifiedEncoderDecoderTransformerTrainingOutput
