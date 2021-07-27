{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GLMHead where

import Control.Monad.Indexed (IxPointed (..), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType, DataType, SDataType)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel)
import Torch.GraduallyTyped.NN.Linear (GLinear, LinearBiasF, LinearWeightF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sGeneratorToDevice)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

data LMHeadHasScaling = LMHeadWithScaling | LMHeadWithoutScaling

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
    { lmHeadInputEmbedDim :: SDim inputEmbedDim,
      lmHeadDense :: dense,
      lmHeadActivation :: activation,
      lmHeadLayerNorm :: layerNorm,
      lmHeadDecoder :: decoder,
      lmHeadBias :: bias,
      lmHeadHasScaling :: LMHeadHasScaling
    } ->
    GLMHead inputEmbedDim dense activation layerNorm decoder bias

type instance
  ModelSpec (GLMHead inputEmbedDim dense activation layerNorm decoder bias) =
    GLMHead inputEmbedDim (ModelSpec dense) (ModelSpec activation) (ModelSpec layerNorm) (ModelSpec decoder) (ModelSpec bias)

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
    ()
  LMHeadBiasF 'ByT5 gradient device dataType vocabDim =
    LMHeadBiasF 'T5 gradient device dataType vocabDim
  LMHeadBiasF 'BART gradient device dataType vocabDim =
    NamedModel (Tensor gradient ('Layout 'Dense) device dataType ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim]))
  LMHeadBiasF 'MBART gradient device dataType vocabDim =
    LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'Pegasus gradient device dataType vocabDim =
    LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'BERT _ _ _ _ =
    ()
  LMHeadBiasF 'RoBERTa gradient device dataType vocabDim =
    LMHeadBiasF 'BERT gradient device dataType vocabDim

lmHeadSpec ::
  forall style gradient device dataType inputEmbedDim vocabDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim inputEmbedDim ->
  SDim vocabDim ->
  Double ->
  GLMHead
    inputEmbedDim
    (LMHeadDenseF style gradient device dataType inputEmbedDim)
    (LMHeadActivationF style)
    (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
    (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
    (LMHeadBiasF style gradient device dataType vocabDim)
lmHeadSpec style gradient device dataType inputEmbedDim vocabDim eps =
  undefined

-- dense ST5 = pure ()
-- dense SByT5 = pure ()
-- dense SBART = pure ()
-- dense SMBART = pure ()
-- dense SPegasus = pure ()
-- dense SBERT = fromStateDict denseSpec (k <> "transform.dense.")
-- dense SRoBERTa = fromStateDict denseSpec (k <> "dense.")
-- dense SGPT2 = undefined
-- activation :: STransformerStyle style -> LMHeadActivationF style
-- activation ST5 = ()
-- activation SByT5 = ()
-- activation SBART = ()
-- activation SMBART = ()
-- activation SPegasus = ()
-- activation SBERT = Gelu
-- activation SRoBERTa = Gelu
-- activation SGPT2 = undefined
-- layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
-- layerNorm ST5 = pure ()
-- layerNorm SByT5 = pure ()
-- layerNorm SBART = pure ()
-- layerNorm SMBART = pure ()
-- layerNorm SPegasus = pure ()
-- layerNorm SBERT = fromStateDict layerNormWithBiasSpec (k <> "transform.LayerNorm.")
-- layerNorm SRoBERTa = fromStateDict layerNormWithBiasSpec (k <> "layer_norm.")
-- layerNorm SGPT2 = undefined
-- decoderWithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType inputEmbedDim vocabDim
-- decoderWithBiasSpec = LinearSpec SWithBias gradient device dataType inputEmbedDim vocabDim
-- decoder ST5 = fromStateDict decoderWithoutBiasSpec k
-- decoder SByT5 = fromStateDict decoderWithoutBiasSpec k
-- decoder SBART = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
-- decoder SMBART = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
-- decoder SPegasus = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
-- decoder SBERT = fromStateDict decoderWithBiasSpec (k <> "decoder.")
-- decoder SRoBERTa = fromStateDict decoderWithBiasSpec (k <> "decoder.")
-- decoder SGPT2 = undefined
-- biasSpec = TensorSpec gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
-- bias ST5 = pure ()
-- bias SByT5 = pure ()
-- bias SBART = fromStateDict biasSpec (k <> "final_logits_bias")
-- bias SMBART = fromStateDict biasSpec (k <> "final_logits_bias")
-- bias SPegasus = fromStateDict biasSpec (k <> "final_logits_bias")
-- bias SBERT = pure ()
-- bias SRoBERTa = pure ()
-- bias SGPT2 = undefined

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
     in runIxStateT (GLMHead <<$>> ireturn inputEmbedDim <<*>> dense <<*>> activation <<*>> layerNorm <<*>> decoder <<*>> bias <<*>> ireturn hasScaling)

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
    output ~ Tensor gradient3 layout3 device3 dataType3 shape3,
    generatorOutputDevice ~ generatorDevice3
  ) =>
  HasForward
    (GLMHead inputEmbedDim dense activation layerNorm decoder ())
    (Tensor gradient layout device dataType shape)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GLMHead {..} input =
    let scaling = (1 :: Double) / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ lmHeadInputEmbedDim)
     in -- bias :: STransformerStyle style -> decoderOutput -> output
        -- bias ST5 = id
        -- bias SByT5 = id
        -- bias SBART = (`add` lmHeadBias)
        -- bias SMBART = (`add` lmHeadBias)
        -- bias SPegasus = (`add` lmHeadBias)
        -- bias SBERT = id
        -- bias SRoBERTa = id
        -- bias SGPT2 = undefined
        runIxStateT $
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

-- >>>= ireturn . bias (sing @style)
