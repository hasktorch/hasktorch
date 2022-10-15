{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module Torch.GraduallyTyped.NN.Transformer.GEncoderOnly where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingKind (fromSing))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GLMHead (GLMHeadF, lmHeadSpec)
import Torch.GraduallyTyped.NN.Transformer.GTransformer (TransformerEncoderF, transformerEncoderSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerHead (..), STransformerStyle (..), TransformerHead (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasDropout, SHasDropout)
import Torch.GraduallyTyped.Prelude (Catch, forgetIsChecked)
import Torch.GraduallyTyped.Prelude.Maybe (SMaybe (SNothing))
import Torch.GraduallyTyped.Prelude.TypeLits (SNat)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient, SGradient)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Prelude hiding (head)

-- | Data type that is used to represent whether the encoder-only transformer model has a scaled embedding.
data EncoderOnlyTransformerHasEmbedScaling
  = EncoderOnlyTransformerWithEmbedScaling
  | EncoderOnlyTransformerWithoutEmbedScaling
  deriving (Eq, Ord, Show, Generic)

type instance ModelSpec EncoderOnlyTransformerHasEmbedScaling = EncoderOnlyTransformerHasEmbedScaling

instance HasInitialize EncoderOnlyTransformerHasEmbedScaling generatorDevice EncoderOnlyTransformerHasEmbedScaling generatorDevice where
  initialize hasEmbedScaling g = pure (hasEmbedScaling, g)

instance HasStateDict EncoderOnlyTransformerHasEmbedScaling where
  fromStateDict hasEmbedScaling _ = pure hasEmbedScaling
  toStateDict _ _ = pure ()

-- | Generic encoder-only transformer model.
-- This is a transformer model that only encodes the input, e.g. BERT.
--
-- - @inputEmbedDim@: the dimension of the input embedding.
-- - @encoder@: a transformer encoder.
-- - @encoderEmbedding@: an embedding layer for the input.
-- - @encoderTypeEmbedding@: an embedding layer for the type of the input.
-- - @head@: a head layer for the output.
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
      eotInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      eotEncoder :: encoder,
      -- | encoder embedding
      eotEmbedding :: encoderEmbedding,
      -- | encoder type embedding
      eotTypeEmbedding :: encoderTypeEmbedding,
      -- | encoder head
      eotHead :: head,
      -- | encoder embedding scaling
      eotEmbedScaling :: EncoderOnlyTransformerHasEmbedScaling
    } ->
    GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head
  deriving stock (Show, Generic)

type instance
  ModelSpec (GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head) =
    GEncoderOnlyTransformer inputEmbedDim (ModelSpec encoder) (ModelSpec encoderEmbedding) (ModelSpec encoderTypeEmbedding) (ModelSpec head)

type family
  GEncoderOnlyTransformerF
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
    (hasDropout :: HasDropout) ::
    Type
  where
  GEncoderOnlyTransformerF style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim hasDropout =
    GEncoderOnlyTransformer
      inputEmbedDim
      (NamedModel (TransformerEncoderF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout))
      (EOTEmbeddingF style gradient device dataType inputEmbedDim vocabDim)
      (EOTTypeEmbeddingF style gradient device dataType inputEmbedDim typeVocabDim)
      (EOTHeadF style transformerHead gradient device dataType inputEmbedDim vocabDim)

-- | Specifies the embedding layer of the encoder-only transformer model.
type family
  EOTEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTEmbeddingF _ gradient device dataType inputEmbedDim vocabDim =
    NamedModel (Embedding gradient ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)

-- | Specifies the type embedding layer of the encoder-only transformer model.
type family
  EOTTypeEmbeddingF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (typeVocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTTypeEmbeddingF 'BERT gradient device dataType inputEmbedDim typeVocabDim =
    NamedModel (Embedding gradient ('Layout 'Dense) device dataType typeVocabDim inputEmbedDim 'Nothing)
  EOTTypeEmbeddingF 'RoBERTa gradient device dataType inputEmbedDim typeVocabDim =
    EOTTypeEmbeddingF 'BERT gradient device dataType inputEmbedDim typeVocabDim

-- | Specifies the head layer of the encoder-only transformer model.
type family
  EOTHeadF
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  EOTHeadF _ 'WithoutHead _ _ _ _ _ =
    ()
  EOTHeadF style 'WithLMHead gradient device dataType inputEmbedDim vocabDim =
    NamedModel (GLMHeadF style gradient device dataType inputEmbedDim vocabDim)

-- | Specifies the parameters of an encoder-only transformer model.
--
-- - @style@: the style of the encoder-only transformer model, e.g. 'SBERT', 'SRoBERTa', etc.
-- - @transformerHead@: the head of the encoder-only transformer model.
-- - @numLayers@: the number of layers of the encoder-only transformer model.
-- - @gradient@: whether to compute the gradient of the model parameters
-- - @device@: the computational device on which the model is allocated.
-- - @dataType@: the data type of the model parameters.
-- - @headDim@: the dimension of all transformer heads in the encoder-only transformer model.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @inputEmbedDim@: the dimension of the input embeddings.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @posEncDim@: the dimension of the positional embeddings.
-- - @vocabDim@: the dimension of the vocabulary.
-- - @typeVocabDim@: the dimension of the type vocabulary.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
encoderOnlyTransformerSpec ::
  forall style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim hasDropout.
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
  SHasDropout hasDropout ->
  Double ->
  Double ->
  ModelSpec (GEncoderOnlyTransformerF style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim hasDropout)
encoderOnlyTransformerSpec style transformerHead numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim typeVocabDim hasDropout dropoutP eps =
  let encoderSpec ST5 = undefined
      encoderSpec SByT5 = undefined
      encoderSpec SBART = undefined
      encoderSpec SMBART = undefined
      encoderSpec SPegasus = undefined
      encoderSpec SBERT = NamedModel "bert." $ encoderSpec' SBERT
      encoderSpec SRoBERTa = NamedModel "roberta." $ encoderSpec' SRoBERTa
      encoderSpec SGPT2 = undefined
      embeddingSpec ST5 = undefined
      embeddingSpec SByT5 = undefined
      embeddingSpec SBART = undefined
      embeddingSpec SMBART = undefined
      embeddingSpec SPegasus = undefined
      embeddingSpec SBERT = NamedModel "bert.embeddings.word_embeddings." embeddingSpec'
      embeddingSpec SRoBERTa = NamedModel "roberta.embeddings.word_embeddings." embeddingSpec'
      embeddingSpec SGPT2 = undefined
      typeEmbeddingSpec ST5 = undefined
      typeEmbeddingSpec SByT5 = undefined
      typeEmbeddingSpec SBART = undefined
      typeEmbeddingSpec SMBART = undefined
      typeEmbeddingSpec SPegasus = undefined
      typeEmbeddingSpec SBERT = NamedModel "bert.embeddings.token_type_embeddings." typeEmbeddingSpec'
      typeEmbeddingSpec SRoBERTa = NamedModel "roberta.embeddings.token_type_embeddings." typeEmbeddingSpec'
      typeEmbeddingSpec SGPT2 = undefined
      headSpec ST5 _ = undefined
      headSpec SByT5 _ = undefined
      headSpec SBART _ = undefined
      headSpec SMBART _ = undefined
      headSpec SPegasus _ = undefined
      headSpec SBERT SWithoutHead = ()
      headSpec SBERT SWithLMHead = NamedModel "cls.predictions." $ headSpec' SBERT
      headSpec SRoBERTa SWithoutHead = ()
      headSpec SRoBERTa SWithLMHead = NamedModel "lm_head." $ headSpec' SRoBERTa
      headSpec SGPT2 _ = undefined
      embedScalingSpec :: STransformerStyle style -> EncoderOnlyTransformerHasEmbedScaling
      embedScalingSpec ST5 = undefined
      embedScalingSpec SByT5 = undefined
      embedScalingSpec SBART = undefined
      embedScalingSpec SMBART = undefined
      embedScalingSpec SPegasus = undefined
      embedScalingSpec SBERT = EncoderOnlyTransformerWithoutEmbedScaling
      embedScalingSpec SRoBERTa = EncoderOnlyTransformerWithoutEmbedScaling
      embedScalingSpec SGPT2 = undefined
   in GEncoderOnlyTransformer
        inputEmbedDim
        (encoderSpec style)
        (embeddingSpec style)
        (typeEmbeddingSpec style)
        (headSpec style transformerHead)
        (embedScalingSpec style)
  where
    encoderSpec' :: _
    encoderSpec' style' = transformerEncoderSpec style' numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim hasDropout dropoutP eps
    embeddingSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType vocabDim inputEmbedDim SNothing
    typeEmbeddingSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType typeVocabDim inputEmbedDim SNothing
    headSpec' :: _
    headSpec' style' = lmHeadSpec style' gradient device dataType inputEmbedDim vocabDim eps

instance
  ( HasInitialize encoder generatorDevice encoder' generatorDevice0,
    HasInitialize encoderEmbedding generatorDevice0 encoderEmbedding' generatorDevice1,
    HasInitialize encoderTypeEmbedding generatorDevice1 encoderTypeEmbedding' generatorDevice2,
    HasInitialize head generatorDevice2 head' generatorOutputDevice
  ) =>
  HasInitialize
    (GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head)
    generatorDevice
    (GEncoderOnlyTransformer inputEmbedDim encoder' encoderEmbedding' encoderTypeEmbedding' head')
    generatorOutputDevice

instance
  (HasStateDict encoder, HasStateDict encoderEmbedding, HasStateDict encoderTypeEmbedding, HasStateDict head) =>
  HasStateDict (GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head)

data
  GSimplifiedEncoderOnlyTransformer
    (model :: Type)
    (mkPos :: Type)
    (mkPaddingMask :: Type)
    (mkAttentionMask :: Type)
  where
  GSimplifiedEncoderOnlyTransformer ::
    forall model mkPos mkPaddingMask mkAttentionMask.
    { -- | encoder-only model
      seotModel :: model,
      -- | make input positions
      seotMkPos :: mkPos,
      -- | make padding mask
      seotMkPaddingMask :: mkPaddingMask,
      -- | make attention mask
      seotMkAttentionMask :: mkAttentionMask
    } ->
    GSimplifiedEncoderOnlyTransformer model mkPos mkPaddingMask mkAttentionMask
  deriving stock (Eq, Ord, Show, Generic)

type instance
  ModelSpec (GSimplifiedEncoderOnlyTransformer model mkPos mkPaddingMask mkAttentionMask) =
    GSimplifiedEncoderOnlyTransformer (ModelSpec model) (ModelSpec mkPos) (ModelSpec mkPaddingMask) (ModelSpec mkAttentionMask)

instance
  ( HasInitialize model generatorDevice model' generatorDevice0,
    HasInitialize mkPos generatorDevice0 mkPos' generatorDevice1,
    HasInitialize mkPaddingMask generatorDevice1 mkPaddingMask' generatorDevice2,
    HasInitialize mkAttentionMask generatorDevice2 mkAttentionMask' generatorOutputDevice
  ) =>
  HasInitialize
    (GSimplifiedEncoderOnlyTransformer model mkPos mkPaddingMask mkAttentionMask)
    generatorDevice
    (GSimplifiedEncoderOnlyTransformer model' mkPos' mkPaddingMask' mkAttentionMask')
    generatorOutputDevice

instance
  (HasStateDict model, HasStateDict mkPos, HasStateDict mkPaddingMask, HasStateDict mkAttentionMask) =>
  HasStateDict (GSimplifiedEncoderOnlyTransformer model mkPos mkPaddingMask mkAttentionMask)

-- | Input data type for use with an encoder-only transformer.
data EncoderOnlyTransformerInput input inputType pos attentionMask where
  EncoderOnlyTransformerInput ::
    forall input inputType pos attentionMask.
    { eotInput :: input,
      eotInputType :: inputType,
      eotPos :: pos,
      eotAttentionMask :: attentionMask
    } ->
    EncoderOnlyTransformerInput input inputType pos attentionMask
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderOnlyTransformerInput input inputType where
  SimplifiedEncoderOnlyTransformerInput ::
    forall input inputType.
    { seotInput :: input,
      seotInputType :: inputType
    } ->
    SimplifiedEncoderOnlyTransformerInput input inputType
  deriving stock (Eq, Ord, Show, Generic)

-- | Output data type for use with an encoder-only transformer.
data EncoderOnlyTransformerOutput output where
  EncoderOnlyTransformerOutput ::
    forall output.
    { eotOutput :: output
    } ->
    EncoderOnlyTransformerOutput output
  deriving stock (Eq, Ord, Show, Generic)

data SimplifiedEncoderOnlyTransformerOutput output paddingMask where
  SimplifiedEncoderOnlyTransformerOutput ::
    forall output paddingMask.
    { seotOutput :: output,
      sedtPaddingMask :: paddingMask
    } ->
    SimplifiedEncoderOnlyTransformerOutput output paddingMask
  deriving stock (Eq, Ord, Show, Generic)

-- | 'HasForward' instance for encoder-only transformers with optional scaling and head.
--
-- @
--    ┌───────┐    ┌───────────┐  ┌─────┐  ┌───────────────┐
--    │ input │    │ inputType │  │ pos │  │ attentionMask │
--    └───┬───┘    └─────┬─────┘  └──┬──┘  └──────┬────────┘
--        │              │           │            │
--        ▼              ▼           │            │
--  eotEmbedding  eotTypeEmbedding   │            │
--        ▼              ▼           │            │
-- (embedScaling)  (embedScaling)    │            │
--        │              │           │            │
--        └────►add◄─────┘           │            │
--               │                   │            │
--               ▼                   │            │
--          eotEncoder◄──────────────┘◄───────────┘
--               ▼
--           (eotHead)
--               │
--               ▼
--          ┌────────┐
--          │ output │
--          └────────┘
-- @
instance
  ( HasForward
      encoderEmbedding
      input
      generatorDevice
      embeddingOutput
      embeddingGeneratorOutputDevice,
    embeddingOutput ~ Tensor gradient' layout' device' dataType' shape',
    HasForward
      encoderTypeEmbedding
      inputType
      embeddingGeneratorOutputDevice
      typeEmbeddingOutput
      typeEmbeddingGeneratorOutputDevice,
    typeEmbeddingOutput ~ Tensor gradient'' layout'' device'' dataType'' shape'',
    HasForward
      encoder
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
      encoderGeneratorOutputDevice,
    Catch (BroadcastShapesF shape' shape''),
    HasForward
      head
      encoderOutput
      encoderGeneratorOutputDevice
      headOutput
      generatorOutputDevice
  ) =>
  HasForward
    (GEncoderOnlyTransformer inputEmbedDim encoder encoderEmbedding encoderTypeEmbedding head)
    (EncoderOnlyTransformerInput input inputType pos attentionMask)
    generatorDevice
    (EncoderOnlyTransformerOutput headOutput)
    generatorOutputDevice
  where
  forward GEncoderOnlyTransformer {..} EncoderOnlyTransformerInput {..} =
    let scaling :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ eotInputEmbedDim
        embeddedInput =
          ireturn eotInput
            >>>= IxStateT . forward eotEmbedding
            >>>= ilift
              . ( \case
                    EncoderOnlyTransformerWithoutEmbedScaling -> pure
                    EncoderOnlyTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                eotEmbedScaling
        embeddedInputType =
          ireturn eotInputType
            >>>= IxStateT . forward eotTypeEmbedding
            >>>= ilift
              . ( \case
                    EncoderOnlyTransformerWithoutEmbedScaling -> pure
                    EncoderOnlyTransformerWithEmbedScaling -> flip mulScalar scaling
                )
                eotEmbedScaling
     in runIxStateT $
          (,) <<$>> embeddedInput <<*>> embeddedInputType
            >>>= ilift . uncurry add
            >>>= (\input' -> IxStateT $ forward eotEncoder (input', eotPos, eotAttentionMask))
            >>>= IxStateT . forward eotHead
            >>>= ireturn . EncoderOnlyTransformerOutput

instance
  ( HasForward
      mkPaddingMask
      input
      generatorDevice
      paddingMask
      generatorDevice,
    HasForward
      mkAttentionMask
      paddingMask
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
      (EncoderOnlyTransformerInput input inputType pos attentionMask)
      generatorDevice
      (EncoderOnlyTransformerOutput output)
      generatorOutputDevice
  ) =>
  HasForward
    (GSimplifiedEncoderOnlyTransformer model mkPos mkPaddingMask mkAttentionMask)
    (SimplifiedEncoderOnlyTransformerInput input inputType)
    generatorDevice
    (SimplifiedEncoderOnlyTransformerOutput output paddingMask)
    generatorOutputDevice
  where
  forward GSimplifiedEncoderOnlyTransformer {..} SimplifiedEncoderOnlyTransformerInput {..} =
    runIxStateT $
      ( let paddingMask = IxStateT . forward seotMkPaddingMask $ seotInput
            pos = IxStateT . forward seotMkPos $ seotInput
         in (,) <<$>> paddingMask <<*>> pos
      )
        >>>= ( \(paddingMask, pos) ->
                 let attentionMask = IxStateT . forward seotMkAttentionMask $ paddingMask
                  in ( EncoderOnlyTransformerInput
                         seotInput
                         seotInputType
                         pos
                         <<$>> attentionMask
                     )
                       >>>= IxStateT . forward seotModel
                       >>>= ( \(EncoderOnlyTransformerOutput output) ->
                                ireturn $ SimplifiedEncoderOnlyTransformerOutput output paddingMask
                            )
             )
