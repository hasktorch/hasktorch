{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.LMHead where

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
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Linear (Linear (..), LinearSpec (..))
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
      lmHeadBias :: bias
    } ->
    GLMHead inputEmbedDim dense activation layerNorm decoder bias

-- | Language modelling head for transformer encoders and decoders.
newtype
  LMHead
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  LMHead ::
    forall style gradient device dataType inputEmbedDim vocabDim.
    GLMHead
      inputEmbedDim
      (LMHeadDenseF style gradient device dataType inputEmbedDim)
      (LMHeadActivationF style)
      (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
      (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
      (LMHeadBiasF style gradient device dataType vocabDim) ->
    LMHead style gradient device dataType inputEmbedDim vocabDim

data
  LMHeadSpec
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  LMHeadSpec ::
    forall style gradient device dataType inputEmbedDim vocabDim.
    STransformerStyle style ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim inputEmbedDim ->
    SDim vocabDim ->
    Double ->
    LMHeadSpec style gradient device dataType inputEmbedDim vocabDim

type instance ModelSpec (LMHead style gradient device dataType inputEmbedDim vocabDim) = LMHeadSpec style gradient device dataType inputEmbedDim vocabDim

type family
  LMHeadDenseF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDenseF 'T5 _ _ _ _ = ()
  LMHeadDenseF 'ByT5 gradient device dataType inputEmbedDim = LMHeadDenseF 'T5 gradient device dataType inputEmbedDim
  LMHeadDenseF 'BART _ _ _ _ = ()
  LMHeadDenseF 'MBART gradient device dataType inputEmbedDim = LMHeadDenseF 'BART gradient device dataType inputEmbedDim
  LMHeadDenseF 'Pegasus gradient device dataType inputEmbedDim = LMHeadDenseF 'BART gradient device dataType inputEmbedDim
  LMHeadDenseF 'BERT gradient device dataType inputEmbedDim = Linear 'WithBias gradient device dataType inputEmbedDim inputEmbedDim
  LMHeadDenseF 'RoBERTa gradient device dataType inputEmbedDim = LMHeadDenseF 'BERT gradient device dataType inputEmbedDim

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
  LMHeadLayerNormF 'ByT5 gradient device dataType inputEmbedDim = LMHeadLayerNormF 'T5 gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'BART _ _ _ _ = ()
  LMHeadLayerNormF 'MBART gradient device dataType inputEmbedDim = LMHeadLayerNormF 'BART gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'Pegasus gradient device dataType inputEmbedDim = LMHeadLayerNormF 'BART gradient device dataType inputEmbedDim
  LMHeadLayerNormF 'BERT gradient device dataType inputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim])
  LMHeadLayerNormF 'RoBERTa gradient device dataType inputEmbedDim = LMHeadLayerNormF 'BERT gradient device dataType inputEmbedDim

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
  LMHeadDecoderF 'T5 gradient device dataType inputEmbedDim vocabDim = Linear 'WithoutBias gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'ByT5 gradient device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'T5 gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim = Linear 'WithoutBias gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'MBART gradient device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'Pegasus gradient device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BART gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BERT gradient device dataType inputEmbedDim vocabDim = Linear 'WithBias gradient device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'RoBERTa gradient device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BERT gradient device dataType inputEmbedDim vocabDim

type family
  LMHeadBiasF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadBiasF 'T5 _ _ _ _ = ()
  LMHeadBiasF 'ByT5 gradient device dataType vocabDim = LMHeadBiasF 'T5 gradient device dataType vocabDim
  LMHeadBiasF 'BART gradient device dataType vocabDim = Tensor gradient ('Layout 'Dense) device dataType ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim])
  LMHeadBiasF 'MBART gradient device dataType vocabDim = LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'Pegasus gradient device dataType vocabDim = LMHeadBiasF 'BART gradient device dataType vocabDim
  LMHeadBiasF 'BERT _ _ _ _ = ()
  LMHeadBiasF 'RoBERTa gradient device dataType vocabDim = LMHeadBiasF 'BERT gradient device dataType vocabDim

instance
  ( dense ~ LMHeadDenseF style gradient device dataType inputEmbedDim,
    HasInitialize dense device dense device,
    activation ~ LMHeadActivationF style,
    HasInitialize activation device activation device,
    layerNorm ~ LMHeadLayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    decoder ~ LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim,
    HasInitialize decoder device decoder device,
    bias ~ LMHeadBiasF style gradient device dataType vocabDim
  ) =>
  HasInitialize
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
    generatorDevice
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
    device
  where
  initialize (LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps) generator =
    let generator' = sGeneratorToDevice device generator
        denseSpec = LinearSpec SWithBias gradient device dataType inputEmbedDim inputEmbedDim
        dense = IxStateT . initialize @dense $
          case style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> denseSpec
            SRoBERTa -> denseSpec
            SGPT2 -> undefined
        activation = IxStateT . initialize @activation $
          case style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> Gelu
            SRoBERTa -> Gelu
            SGPT2 -> undefined
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $
          case style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> layerNormWithBiasSpec
            SRoBERTa -> layerNormWithBiasSpec
            SGPT2 -> undefined
        decoderWithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType inputEmbedDim vocabDim
        decoderWithBiasSpec = LinearSpec SWithBias gradient device dataType inputEmbedDim vocabDim
        decoder = IxStateT . initialize @decoder $
          case style of
            ST5 -> decoderWithoutBiasSpec
            SByT5 -> decoderWithoutBiasSpec
            SBART -> decoderWithoutBiasSpec
            SMBART -> decoderWithoutBiasSpec
            SPegasus -> decoderWithoutBiasSpec
            SBERT -> decoderWithBiasSpec
            SRoBERTa -> decoderWithBiasSpec
            SGPT2 -> undefined
        biasSpec = TensorSpec gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
        bias = ireturn $
          case style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> sZeros biasSpec
            SMBART -> sZeros biasSpec
            SPegasus -> sZeros biasSpec
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
     in runIxStateT
          ( (GLMHead <<$>> ireturn inputEmbedDim <<*>> dense <<*>> activation <<*>> layerNorm <<*>> decoder <<*>> bias)
              >>>= ireturn . LMHead
          )
          generator'

instance
  SingI style =>
  HasStateDict
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
  where
  fromStateDict (LMHeadSpec style gradient device dataType inputEmbedDim vocabDim eps) k =
    let denseSpec = LinearSpec SWithBias gradient device dataType inputEmbedDim inputEmbedDim
        dense ST5 = pure ()
        dense SByT5 = pure ()
        dense SBART = pure ()
        dense SMBART = pure ()
        dense SPegasus = pure ()
        dense SBERT = fromStateDict denseSpec (k <> "transform.dense.")
        dense SRoBERTa = fromStateDict denseSpec (k <> "dense.")
        dense SGPT2 = undefined
        activation :: STransformerStyle style -> LMHeadActivationF style
        activation ST5 = ()
        activation SByT5 = ()
        activation SBART = ()
        activation SMBART = ()
        activation SPegasus = ()
        activation SBERT = Gelu
        activation SRoBERTa = Gelu
        activation SGPT2 = undefined
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNorm ST5 = pure ()
        layerNorm SByT5 = pure ()
        layerNorm SBART = pure ()
        layerNorm SMBART = pure ()
        layerNorm SPegasus = pure ()
        layerNorm SBERT = fromStateDict layerNormWithBiasSpec (k <> "transform.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict layerNormWithBiasSpec (k <> "layer_norm.")
        layerNorm SGPT2 = undefined
        decoderWithoutBiasSpec = LinearSpec SWithoutBias gradient device dataType inputEmbedDim vocabDim
        decoderWithBiasSpec = LinearSpec SWithBias gradient device dataType inputEmbedDim vocabDim
        decoder ST5 = fromStateDict decoderWithoutBiasSpec k
        decoder SByT5 = fromStateDict decoderWithoutBiasSpec k
        decoder SBART = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
        decoder SMBART = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
        decoder SPegasus = fromStateDict decoderWithoutBiasSpec (k <> "lm_head.")
        decoder SBERT = fromStateDict decoderWithBiasSpec (k <> "decoder.")
        decoder SRoBERTa = fromStateDict decoderWithBiasSpec (k <> "decoder.")
        decoder SGPT2 = undefined
        biasSpec = TensorSpec gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
        bias ST5 = pure ()
        bias SByT5 = pure ()
        bias SBART = fromStateDict biasSpec (k <> "final_logits_bias")
        bias SMBART = fromStateDict biasSpec (k <> "final_logits_bias")
        bias SPegasus = fromStateDict biasSpec (k <> "final_logits_bias")
        bias SBERT = pure ()
        bias SRoBERTa = pure ()
        bias SGPT2 = undefined
     in LMHead
          <$> ( GLMHead inputEmbedDim
                  <$> dense style
                  <*> pure (activation style)
                  <*> layerNorm style
                  <*> decoder style
                  <*> bias style
              )
  toStateDict k (LMHead GLMHead {..}) =
    let dense ST5 = const $ pure ()
        dense SByT5 = const $ pure ()
        dense SBART = const $ pure ()
        dense SMBART = const $ pure ()
        dense SPegasus = const $ pure ()
        dense SBERT = toStateDict (k <> "transform.dense.")
        dense SRoBERTa = toStateDict (k <> "dense.")
        dense SGPT2 = undefined
        layerNorm ST5 = const $ pure ()
        layerNorm SByT5 = const $ pure ()
        layerNorm SBART = const $ pure ()
        layerNorm SMBART = const $ pure ()
        layerNorm SPegasus = const $ pure ()
        layerNorm SBERT = toStateDict (k <> "transform.LayerNorm.")
        layerNorm SRoBERTa = toStateDict (k <> "layer_norm.")
        layerNorm SGPT2 = undefined
        decoder ST5 = toStateDict k
        decoder SByT5 = toStateDict k
        decoder SBART = toStateDict (k <> "lm_head.")
        decoder SMBART = toStateDict (k <> "lm_head.")
        decoder SPegasus = toStateDict (k <> "lm_head.")
        decoder SBERT = toStateDict (k <> "decoder.")
        decoder SRoBERTa = toStateDict (k <> "decoder.")
        decoder SGPT2 = undefined
        bias ST5 = const $ pure ()
        bias SByT5 = const $ pure ()
        bias SBART = toStateDict (k <> "final_logits_bias")
        bias SMBART = toStateDict (k <> "final_logits_bias")
        bias SPegasus = toStateDict (k <> "final_logits_bias")
        bias SBERT = const $ pure ()
        bias SRoBERTa = const $ pure ()
        bias SGPT2 = undefined
     in do
          () <- dense (sing @style) lmHeadDense
          () <- layerNorm (sing @style) lmHeadLayerNorm
          () <- decoder (sing @style) lmHeadDecoder
          () <- bias (sing @style) lmHeadBias
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
  ( SingI style,
    HasForward
      (LMHeadDenseF style gradient device dataType inputEmbedDim)
      input
      generatorDevice
      denseOutput
      denseGeneratorOutputDevice,
    HasForward
      (LMHeadActivationF style)
      denseOutput
      denseGeneratorOutputDevice
      activationOutput
      activationGeneratorOutputDevice,
    HasForward
      (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
      activationOutput
      activationGeneratorOutputDevice
      layerNormOutput
      layerNormGeneratorOutputDevice,
    HasForward
      (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
      layerNormOutput
      layerNormGeneratorOutputDevice
      decoderOutput
      generatorOutputDevice,
    decoderOutput ~ Tensor gradient' layout' device' dataType' shape',
    output ~ LMHeadOutputF style decoderOutput gradient device dataType vocabDim
  ) =>
  HasForward
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
    input
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (LMHead GLMHead {..}) input =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ lmHeadInputEmbedDim
        scaling :: STransformerStyle style -> decoderOutput -> decoderOutput
        scaling ST5 = flip divScalar s
        scaling SByT5 = flip divScalar s
        scaling SBART = id
        scaling SMBART = id
        scaling SPegasus = id
        scaling SBERT = id
        scaling SRoBERTa = id
        scaling SGPT2 = undefined
        bias :: STransformerStyle style -> decoderOutput -> output
        bias ST5 = id
        bias SByT5 = id
        bias SBART = (`add` lmHeadBias)
        bias SMBART = (`add` lmHeadBias)
        bias SPegasus = (`add` lmHeadBias)
        bias SBERT = id
        bias SRoBERTa = id
        bias SGPT2 = undefined
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward lmHeadDense
            >>>= IxStateT . forward lmHeadActivation
            >>>= IxStateT . forward lmHeadLayerNorm
            >>>= IxStateT . forward lmHeadDecoder
            >>>= ireturn . scaling (sing @style)
            >>>= ireturn . bias (sing @style)
