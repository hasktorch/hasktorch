{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GLMHead where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType, DataType, SDataType (..))
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Linear (GLinear, LinearBiasF, LinearWeightF, linearSpec)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | A data type that represents whether or not the language modelling head
-- has a scaled decoder output.
data LMHeadHasScaling
  = LMHeadWithScaling
  | LMHeadWithoutScaling

-- | Generic language modelling head for transformer encoders and decoders.
--
-- - @inputEmbedDim@ is the dimension of the input embedding.
-- - @dense@ is a dense layer.
-- - @activation@ is an activation function.
-- - @layerNorm@ is a layer normalization layer.
-- - @decoder@ is a decoder layer.
-- - @bias@ is a bias layer.
data
  GLMHead
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dense :: Type)
    (activation :: Type)
    (layerNorm :: Type)
    (decoder :: Type)
    (bias :: Type)
  where
  GLMHead ::
    forall inputEmbedDim dense activation layerNorm decoder bias.
    { -- | the dimension of the input embedding.
      lmHeadInputEmbedDim :: SDim inputEmbedDim,
      -- | the dense layer.
      lmHeadDense :: dense,
      -- | the activation function.
      lmHeadActivation :: activation,
      -- | the layer normalization layer.
      lmHeadLayerNorm :: layerNorm,
      -- | the decoder layer.
      lmHeadDecoder :: decoder,
      -- | the bias layer.
      lmHeadBias :: bias,
      -- | whether or not the head has a scaled decoder output.
      lmHeadHasScaling :: LMHeadHasScaling
    } ->
    GLMHead inputEmbedDim dense activation layerNorm decoder bias

type instance
  ModelSpec (GLMHead inputEmbedDim dense activation layerNorm decoder bias) =
    GLMHead inputEmbedDim (ModelSpec dense) (ModelSpec activation) (ModelSpec layerNorm) (ModelSpec decoder) (ModelSpec bias)

-- | Generic data type for biasing the language model head.
data GBias (bias :: Type) where GBias :: forall bias. bias -> GBias bias

type instance ModelSpec (GBias bias) = GBias (ModelSpec bias)

-- | Specifies the dense layer of the language model head.
type family
  LMHeadDenseF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDenseF 'T5 _ _ _ _ =
    ()
  LMHeadDenseF 'ByT5 gradient device dataType inputEmbedDim =
    LMHeadDenseF 'T5 gradient device dataType inputEmbedDim
  LMHeadDenseF 'BART _ _ _ _ =
    ()
  LMHeadDenseF 'MBART gradient device dataType inputEmbedDim =
    LMHeadDenseF 'BART gradient device dataType inputEmbedDim
  LMHeadDenseF 'Pegasus gradient device dataType inputEmbedDim =
    LMHeadDenseF 'BART gradient device dataType inputEmbedDim
  LMHeadDenseF 'BERT gradient device dataType inputEmbedDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType inputEmbedDim inputEmbedDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType inputEmbedDim))
      )
  LMHeadDenseF 'RoBERTa gradient device dataType inputEmbedDim =
    LMHeadDenseF 'BERT gradient device dataType inputEmbedDim

-- | Specifies the activation function of the language model head.
type family
  LMHeadActivationF
    (style :: TransformerStyle) ::
    Type
  where
  LMHeadActivationF 'T5 = ()
  LMHeadActivationF 'ByT5 = LMHeadActivationF 'T5
  LMHeadActivationF 'BART = ()
  LMHeadActivationF 'MBART = LMHeadActivationF 'BART
  LMHeadActivationF 'Pegasus = LMHeadActivationF 'BART
  LMHeadActivationF 'BERT = Gelu
  LMHeadActivationF 'RoBERTa = LMHeadActivationF 'BERT

-- | Specifies the layer normalization layer of the language model head.
type family
  LMHeadLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadLayerNormF 'T5 _ _ _ _ = ()
  LMHeadLayerNormF 'ByT5 gradient device dataType inputEmbedDim =
    LMHeadLayerNormF 'T5 gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'BART _ _ _ _ = ()
  LMHeadLayerNormF 'MBART gradient device dataType inputEmbedDim =
    LMHeadLayerNormF 'BART gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'Pegasus gradient device dataType inputEmbedDim =
    LMHeadLayerNormF 'BART gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'BERT gradient device dataType inputEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))
  LMHeadLayerNormF 'RoBERTa gradient device dataType inputEmbedDim =
    LMHeadLayerNormF 'BERT gradient device dataType inputEmbedDim

-- | Specifies the decoder layer of the language model head.
type family
  LMHeadDecoderF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDecoderF 'T5 gradient device dataType inputEmbedDim vocabDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType inputEmbedDim vocabDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType vocabDim))
      )
  LMHeadDecoderF 'ByT5 gradient device dataType inputEmbedDim vocabDim =
    LMHeadDecoderF 'T5 gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType inputEmbedDim vocabDim))
          (NamedModel (LinearBiasF 'WithoutBias gradient device dataType vocabDim))
      )
  LMHeadDecoderF 'MBART gradient device dataType inputEmbedDim vocabDim =
    LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'Pegasus gradient device dataType inputEmbedDim vocabDim =
    LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BERT gradient device dataType inputEmbedDim vocabDim =
    NamedModel
      ( GLinear
          (NamedModel (LinearWeightF gradient device dataType inputEmbedDim vocabDim))
          (NamedModel (LinearBiasF 'WithBias gradient device dataType vocabDim))
      )
  LMHeadDecoderF 'RoBERTa gradient device dataType inputEmbedDim vocabDim =
    LMHeadDecoderF 'BERT gradient device dataType inputEmbedDim vocabDim

-- | Specifies the bias layer of the language model head.
type family
  LMHeadBiasF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadBiasF 'T5 _ _ _ _ =
    GBias ()
  LMHeadBiasF 'ByT5 gradient device dataType vocabDim =
    LMHeadBiasF 'T5 gradient device dataType vocabDim
  LMHeadBiasF 'BART gradient device dataType vocabDim =
    GBias (NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim])))
  LMHeadBiasF 'MBART gradient device dataType vocabDim =
    LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'Pegasus gradient device dataType vocabDim =
    LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'BERT _ _ _ _ =
    GBias ()
  LMHeadBiasF 'RoBERTa gradient device dataType vocabDim =
    LMHeadBiasF 'BERT gradient device dataType vocabDim

-- |
lmHeadSpec ::
  forall style gradient device dataType inputEmbedDim vocabDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim inputEmbedDim ->
  SDim vocabDim ->
  Double ->
  ModelSpec
    ( GLMHead
        inputEmbedDim
        (LMHeadDenseF style gradient device dataType inputEmbedDim)
        (LMHeadActivationF style)
        (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
        (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
        (LMHeadBiasF style gradient device dataType vocabDim)
    )
lmHeadSpec style gradient device dataType inputEmbedDim vocabDim eps =
  let denseSpec ST5 = ()
      denseSpec SByT5 = ()
      denseSpec SBART = ()
      denseSpec SMBART = ()
      denseSpec SPegasus = ()
      denseSpec SBERT = NamedModel "transform.dense." linearSpec'
      denseSpec SRoBERTa = NamedModel "dense." linearSpec'
      denseSpec SGPT2 = undefined
      activationSpec :: STransformerStyle style -> ModelSpec (LMHeadActivationF style)
      activationSpec ST5 = ()
      activationSpec SByT5 = ()
      activationSpec SBART = ()
      activationSpec SMBART = ()
      activationSpec SPegasus = ()
      activationSpec SBERT = Gelu
      activationSpec SRoBERTa = Gelu
      activationSpec SGPT2 = undefined
      layerNormSpec ST5 = ()
      layerNormSpec SByT5 = ()
      layerNormSpec SBART = ()
      layerNormSpec SMBART = ()
      layerNormSpec SPegasus = ()
      layerNormSpec SBERT = NamedModel "transform.LayerNorm." layerNormSpec'
      layerNormSpec SRoBERTa = NamedModel "layer_norm." layerNormSpec'
      layerNormSpec SGPT2 = undefined
      decoderSpec ST5 = NamedModel mempty linearWithoutBiasSpec'
      decoderSpec SByT5 = NamedModel mempty linearWithoutBiasSpec'
      decoderSpec SBART = NamedModel "lm_head." linearWithoutBiasSpec'
      decoderSpec SMBART = NamedModel "lm_head." linearWithoutBiasSpec'
      decoderSpec SPegasus = NamedModel "lm_head." linearWithoutBiasSpec'
      decoderSpec SBERT = NamedModel "decoder." linearWithBiasSpec'
      decoderSpec SRoBERTa = NamedModel "decoder." linearWithBiasSpec'
      decoderSpec SGPT2 = undefined
      biasSpec ST5 = GBias ()
      biasSpec SByT5 = GBias ()
      biasSpec SBART = GBias (NamedModel "final_logits_bias" biasSpec')
      biasSpec SMBART = GBias (NamedModel "final_logits_bias" biasSpec')
      biasSpec SPegasus = GBias (NamedModel "final_logits_bias" biasSpec')
      biasSpec SBERT = GBias ()
      biasSpec SRoBERTa = GBias ()
      biasSpec SGPT2 = undefined
      scalingSpec :: STransformerStyle style -> LMHeadHasScaling
      scalingSpec ST5 = LMHeadWithScaling
      scalingSpec SByT5 = LMHeadWithScaling
      scalingSpec SBART = LMHeadWithoutScaling
      scalingSpec SMBART = LMHeadWithoutScaling
      scalingSpec SPegasus = LMHeadWithoutScaling
      scalingSpec SBERT = LMHeadWithoutScaling
      scalingSpec SRoBERTa = LMHeadWithoutScaling
      scalingSpec SGPT2 = undefined
   in GLMHead inputEmbedDim (denseSpec style) (activationSpec style) (layerNormSpec style) (decoderSpec style) (biasSpec style) (scalingSpec style)
  where
    linearSpec' = linearSpec SWithBias gradient device dataType inputEmbedDim inputEmbedDim
    biasSpec' = TensorSpec gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
    layerNormSpec' = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
    linearWithoutBiasSpec' = linearSpec SWithoutBias gradient device dataType inputEmbedDim vocabDim
    linearWithBiasSpec' = linearSpec SWithBias gradient device dataType inputEmbedDim vocabDim

instance
  ( HasInitialize dense generatorDevice dense generatorDevice,
    HasInitialize activation generatorDevice activation generatorDevice,
    HasInitialize layerNorm generatorDevice layerNorm generatorDevice,
    HasInitialize decoder generatorDevice decoder generatorDevice,
    HasInitialize bias generatorDevice bias generatorDevice
  ) =>
  HasInitialize
    (GLMHead inputEmbedDim dense activation layerNorm decoder bias)
    generatorDevice
    (GLMHead inputEmbedDim dense activation layerNorm decoder bias)
    generatorDevice
  where
  initialize (GLMHead inputEmbedDim denseSpec activationSpec layerNormSpec decoderSpec biasSpec hasScaling) =
    let dense = IxStateT . initialize $ denseSpec
        activation = IxStateT . initialize $ activationSpec
        layerNorm = IxStateT . initialize $ layerNormSpec
        decoder = IxStateT . initialize $ decoderSpec
        bias = IxStateT . initialize $ biasSpec
     in runIxStateT
          ( GLMHead inputEmbedDim
              <<$>> dense
              <<*>> activation
              <<*>> layerNorm
              <<*>> decoder
              <<*>> bias
              <<*>> ireturn hasScaling
          )

instance HasInitialize (GBias ()) generatorDevice (GBias ()) generatorDevice where
  initialize (GBias ()) g = pure (GBias (), g)

instance
  HasInitialize
    (GBias (Tensor biasGradient biasLayout biasDevice biasDataType biasShape))
    generatorDevice
    (GBias (Tensor biasGradient biasLayout biasDevice biasDataType biasShape))
    generatorDevice
  where
  initialize (GBias biasSpec) =
    runIxStateT (GBias <<$>> (ireturn . sZeros $ biasSpec))

instance
  HasInitialize
    (GBias (NamedModel (Tensor biasGradient biasLayout biasDevice biasDataType biasShape)))
    generatorDevice
    (GBias (NamedModel (Tensor biasGradient biasLayout biasDevice biasDataType biasShape)))
    generatorDevice
  where
  initialize (GBias (NamedModel biasName biasSpec)) =
    runIxStateT (GBias <<$>> (ireturn . NamedModel biasName . sZeros $ biasSpec))

instance
  ( HasStateDict dense,
    HasStateDict activation,
    HasStateDict layerNorm,
    HasStateDict decoder,
    HasStateDict bias
  ) =>
  HasStateDict (GLMHead inputEmbedDim dense activation layerNorm decoder bias)
  where
  fromStateDict (GLMHead inputEmbedDim denseSpec activationSpec layerNormSpec decoderSpec biasSpec hasScaling) k =
    GLMHead inputEmbedDim
      <$> fromStateDict denseSpec k
      <*> fromStateDict activationSpec k
      <*> fromStateDict layerNormSpec k
      <*> fromStateDict decoderSpec k
      <*> fromStateDict biasSpec k
      <*> pure hasScaling
  toStateDict k GLMHead {..} = do
    () <- toStateDict k lmHeadDense
    () <- toStateDict k lmHeadActivation
    () <- toStateDict k lmHeadLayerNorm
    () <- toStateDict k lmHeadDecoder
    () <- toStateDict k lmHeadBias
    pure ()

instance HasStateDict model => HasStateDict (GBias model) where
  fromStateDict (GBias biasSpec) k = GBias <$> fromStateDict biasSpec k
  toStateDict k (GBias bias) = toStateDict k bias

type family
  LMHeadOutputF
    (style :: TransformerStyle)
    (decoderOutput :: Type)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadOutputF 'T5 decoderOutput _ _ _ _ = decoderOutput
  LMHeadOutputF 'ByT5 decoderOutput gradient device dataType vocabDim = LMHeadOutputF 'T5 decoderOutput gradient device dataType vocabDim
  LMHeadOutputF 'BART (Tensor gradient' layout' device' dataType' shape') gradient device dataType vocabDim =
    Tensor
      (gradient' <|> gradient)
      (layout' <+> 'Layout 'Dense)
      (device' <+> device)
      (dataType' <+> dataType)
      (BroadcastShapesF shape' ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim]))
  LMHeadOutputF 'MBART decoderOutput gradient device dataType vocabDim = LMHeadOutputF 'BART decoderOutput gradient device dataType vocabDim
  LMHeadOutputF 'Pegasus decoderOutput gradient device dataType vocabDim = LMHeadOutputF 'BART decoderOutput gradient device dataType vocabDim
  LMHeadOutputF 'RoBERTa decoderOutput _ _ _ _ = decoderOutput
  LMHeadOutputF 'BERT decoderOutput _ _ _ _ = decoderOutput

-- | 'HasForward' instance for 'LMHead'.
--
-- @
--     ┌───────┐
--     │ input │
--     └───┬───┘
--         │
--         ▼
--   (lmHeadDense)
--         ▼
-- (lmHeadActivation)
--         ▼
-- (lmHeadLayerNorm)
--         ▼
--   lmHeadDecoder
--         ▼
--     (scaling)
--         ▼
--    (lmHeadBias)
--         │
--         ▼
-- ┌───────────────┐
-- │ decoderOutput │
-- └───────────────┘
-- @
instance
  ( HasForward
      dense
      (Tensor gradient layout device dataType shape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      activation
      tensor0
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      layerNorm
      tensor1
      generatorDevice1
      tensor2
      generatorDevice2,
    HasForward
      decoder
      tensor2
      generatorDevice2
      (Tensor gradient3 layout3 device3 dataType3 shape3)
      generatorDevice3,
    HasForward
      bias
      (Tensor gradient3 layout3 device3 dataType3 shape3)
      generatorDevice3
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GLMHead inputEmbedDim dense activation layerNorm decoder bias)
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GLMHead {..} input =
    let scaling = (1 :: Double) / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ lmHeadInputEmbedDim)
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward lmHeadDense
            >>>= IxStateT . forward lmHeadActivation
            >>>= IxStateT . forward lmHeadLayerNorm
            >>>= IxStateT . forward lmHeadDecoder
            >>>= ireturn
              . ( \case
                    LMHeadWithoutScaling -> id
                    LMHeadWithScaling -> flip mulScalar scaling
                )
                lmHeadHasScaling
            >>>= IxStateT . forward lmHeadBias

instance
  HasForward
    (GBias ())
    (Tensor gradient layout device dataType shape)
    generatorDevice
    (Tensor gradient layout device dataType shape)
    generatorDevice
  where
  forward (GBias bias) = forward bias

instance
  ( output
      ~ Tensor
          (gradient <|> biasGradient)
          (layout <+> biasLayout)
          (device <+> biasDevice)
          (dataType <+> biasDataType)
          (BroadcastShapesF shape biasShape)
  ) =>
  HasForward
    (GBias (Tensor biasGradient biasLayout biasDevice biasDataType biasShape))
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorDevice
  where
  forward (GBias bias) input g = pure (input `add` bias, g)

instance
  ( output
      ~ Tensor
          (gradient <|> biasGradient)
          (layout <+> biasLayout)
          (device <+> biasDevice)
          (dataType <+> biasDataType)
          (BroadcastShapesF shape biasShape)
  ) =>
  HasForward
    (GBias (NamedModel (Tensor biasGradient biasLayout biasDevice biasDataType biasShape)))
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorDevice
  where
  forward (GBias (NamedModel _ bias)) input g = pure (input `add` bias, g)
