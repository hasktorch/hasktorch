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
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type, Constraint)
import Data.Singletons (SingI (..), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device, DeviceType, KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

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
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  LMHead ::
    forall style device dataType inputEmbedDim vocabDim.
    GLMHeadF style device dataType inputEmbedDim vocabDim ->
    LMHead style device dataType inputEmbedDim vocabDim

type GLMHeadF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (vocabDim :: Dim (Name Symbol) (Size Nat)) =
  GLMHead
    inputEmbedDim
    (LMHeadDenseF style device dataType inputEmbedDim)
    (LMHeadActivationF style)
    (LMHeadLayerNormF style device dataType inputEmbedDim)
    (LMHeadDecoderF style device dataType inputEmbedDim vocabDim)
    (LMHeadBiasF style device dataType vocabDim)

type family
  LMHeadDenseF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDenseF 'T5 _ _ _ = ()
  LMHeadDenseF 'ByT5 device dataType inputEmbedDim = LMHeadDenseF 'T5 device dataType inputEmbedDim
  LMHeadDenseF 'BART _ _ _ = ()
  LMHeadDenseF 'MBART device dataType inputEmbedDim = LMHeadDenseF 'BART device dataType inputEmbedDim
  LMHeadDenseF 'Pegasus device dataType inputEmbedDim = LMHeadDenseF 'BART device dataType inputEmbedDim
  LMHeadDenseF 'BERT device dataType inputEmbedDim = Linear 'WithBias device dataType inputEmbedDim inputEmbedDim
  LMHeadDenseF 'RoBERTa device dataType inputEmbedDim = LMHeadDenseF 'BERT device dataType inputEmbedDim

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
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadLayerNormF 'T5 _ _ _ = ()
  LMHeadLayerNormF 'ByT5 device dataType inputEmbedDim = LMHeadLayerNormF 'T5 device dataType inputEmbedDim
  LMHeadLayerNormF 'BART _ _ _ = ()
  LMHeadLayerNormF 'MBART device dataType inputEmbedDim = LMHeadLayerNormF 'BART device dataType inputEmbedDim
  LMHeadLayerNormF 'Pegasus device dataType inputEmbedDim = LMHeadLayerNormF 'BART device dataType inputEmbedDim
  LMHeadLayerNormF 'BERT device dataType inputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  LMHeadLayerNormF 'RoBERTa device dataType inputEmbedDim = LMHeadLayerNormF 'BERT device dataType inputEmbedDim

type family
  LMHeadDecoderF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDecoderF 'T5 device dataType inputEmbedDim vocabDim = Linear 'WithoutBias device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'ByT5 device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'T5 device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BART device dataType inputEmbedDim vocabDim = Linear 'WithoutBias device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'MBART device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BART device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'Pegasus device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BART device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'BERT device dataType inputEmbedDim vocabDim = Linear 'WithBias device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'RoBERTa device dataType inputEmbedDim vocabDim = LMHeadDecoderF 'BERT device dataType inputEmbedDim vocabDim

type family
  LMHeadBiasF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadBiasF 'T5 _ _ _ = ()
  LMHeadBiasF 'ByT5 device dataType vocabDim = LMHeadBiasF 'T5 device dataType vocabDim
  LMHeadBiasF 'BART device dataType vocabDim = Tensor 'WithGradient ('Layout 'Dense) device dataType ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim])
  LMHeadBiasF 'MBART device dataType vocabDim = LMHeadBiasF 'BART device dataType vocabDim
  LMHeadBiasF 'Pegasus device dataType vocabDim = LMHeadBiasF 'BART device dataType vocabDim
  LMHeadBiasF 'BERT _ _ _ = ()
  LMHeadBiasF 'RoBERTa device dataType vocabDim = LMHeadBiasF 'BERT device dataType vocabDim

instance
  ( SingI style,
    dense ~ LMHeadDenseF style device dataType inputEmbedDim,
    HasInitialize dense input generator generator',
    activation ~ LMHeadActivationF style,
    HasInitialize activation input' generator' generator'',
    layerNorm ~ LMHeadLayerNormF style device dataType inputEmbedDim,
    HasInitialize layerNorm input'' generator'' generator''',
    decoder ~ LMHeadDecoderF style device dataType inputEmbedDim vocabDim,
    HasInitialize decoder input''' generator''' generator'''',
    bias ~ LMHeadBiasF style device dataType vocabDim
  ) =>
  HasInitialize
    (LMHead style device dataType inputEmbedDim vocabDim)
    (SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim, Double)
    generator
    generator''''
  where
  initialize (device, dataType, inputEmbedDim, vocabDim, eps) =
    let dense = IxState @generator $
          case sing @style of
            ST5 -> initialize ()
            SByT5 -> initialize ()
            SBART -> initialize ()
            SMBART -> initialize ()
            SPegasus -> initialize ()
            SBERT -> initialize (device, dataType, inputEmbedDim, inputEmbedDim)
            SRoBERTa -> initialize (device, dataType, inputEmbedDim, inputEmbedDim)
            SGPT2 -> undefined
        activation = IxState @generator' $
          case sing @style of
            ST5 -> initialize ()
            SByT5 -> initialize ()
            SBART -> initialize ()
            SMBART -> initialize ()
            SPegasus -> initialize ()
            SBERT -> initialize ()
            SRoBERTa -> initialize ()
            SGPT2 -> undefined
        layerNorm = IxState @generator'' $
          case sing @style of
            ST5 -> initialize ()
            SByT5 -> initialize ()
            SBART -> initialize ()
            SMBART -> initialize ()
            SPegasus -> initialize ()
            SBERT -> initialize (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
            SRoBERTa -> initialize (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
            SGPT2 -> undefined
        decoder = IxState @generator''' $
          case sing @style of
            ST5 -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SByT5 -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SBART -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SMBART -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SPegasus -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SBERT -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SRoBERTa -> initialize (device, dataType, inputEmbedDim, vocabDim)
            SGPT2 -> undefined
        bias = ireturn $
          case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> sZeros SWithGradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SMBART -> sZeros SWithGradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SPegasus -> sZeros SWithGradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
     in runIxState $
          (GLMHead <<$>> ireturn inputEmbedDim <<*>> dense <<*>> activation <<*>> layerNorm <<*>> decoder <<*>> bias)
            >>>= ireturn . LMHead

lookupLMHead ::
  forall style device dataType inputEmbedDim vocabDim m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim inputEmbedDim,
    KnownDim vocabDim
  ) =>
  SDim inputEmbedDim ->
  Double ->
  String ->
  m (LMHead style device dataType inputEmbedDim vocabDim)
lookupLMHead inputEmbedDim eps prefix =
  let dense ST5 = pure ()
      dense SByT5 = pure ()
      dense SBART = pure ()
      dense SMBART = pure ()
      dense SPegasus = pure ()
      dense SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "transform.dense.weight")
          <*> lookupTensor (prefix <> "transform.dense.bias")
      dense SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "dense.weight")
          <*> lookupTensor (prefix <> "dense.bias")
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
      layerNorm ST5 = pure ()
      layerNorm SByT5 = pure ()
      layerNorm SBART = pure ()
      layerNorm SMBART = pure ()
      layerNorm SPegasus = pure ()
      layerNorm SBERT =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "transform.LayerNorm.weight")
          <*> lookupTensor (prefix <> "transform.LayerNorm.bias")
          <*> pure eps
      layerNorm SRoBERTa =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> lookupTensor (prefix <> "layer_norm.bias")
          <*> pure eps
      layerNorm SGPT2 = undefined
      decoder ST5 = LinearWithoutBias <$> lookupTensor (prefix <> "weight")
      decoder SByT5 = LinearWithoutBias <$> lookupTensor (prefix <> "weight")
      decoder SBART = LinearWithoutBias <$> lookupTensor (prefix <> "lm_head.weight")
      decoder SMBART = LinearWithoutBias <$> lookupTensor (prefix <> "lm_head.weight")
      decoder SPegasus = LinearWithoutBias <$> lookupTensor (prefix <> "lm_head.weight")
      decoder SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "decoder.weight")
          <*> lookupTensor (prefix <> "decoder.bias")
      decoder SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "decoder.weight")
          <*> lookupTensor (prefix <> "decoder.bias")
      decoder SGPT2 = undefined
      bias ST5 = pure ()
      bias SByT5 = pure ()
      bias SBART = lookupTensor (prefix <> "final_logits_bias")
      bias SMBART = lookupTensor (prefix <> "final_logits_bias")
      bias SPegasus = lookupTensor (prefix <> "final_logits_bias")
      bias SBERT = pure ()
      bias SRoBERTa = pure ()
      bias SGPT2 = undefined
   in LMHead
        <$> ( GLMHead
                <$> pure inputEmbedDim
                <*> dense (sing @style)
                <*> pure (activation $ sing @style)
                <*> layerNorm (sing @style)
                <*> decoder (sing @style)
                <*> bias (sing @style)
            )

type family
  LMHeadOutputF
    (style :: TransformerStyle)
    (decoderOutput :: Type)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadOutputF 'T5 decoderOutput _ _ _ = decoderOutput
  LMHeadOutputF 'ByT5 decoderOutput device dataType vocabDim = LMHeadOutputF 'T5 decoderOutput device dataType vocabDim
  LMHeadOutputF 'BART (Tensor requiresGradient' layout' device' dataType' shape') device dataType vocabDim =
    Tensor
      'WithGradient
      (layout' <+> 'Layout 'Dense)
      (device' <+> device)
      (dataType' <+> dataType)
      (BroadcastShapesF shape' ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim]))
  LMHeadOutputF 'MBART decoderOutput device dataType vocabDim = LMHeadOutputF 'BART decoderOutput device dataType vocabDim
  LMHeadOutputF 'Pegasus decoderOutput device dataType vocabDim = LMHeadOutputF 'BART decoderOutput device dataType vocabDim
  LMHeadOutputF 'RoBERTa decoderOutput _ _ _ = decoderOutput
  LMHeadOutputF 'BERT decoderOutput _ _ _ = decoderOutput

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
      (LMHeadDenseF style device dataType inputEmbedDim)
      input
      generator
      denseOutput
      denseGeneratorOutput,
    HasForward
      (LMHeadActivationF style)
      denseOutput
      denseGeneratorOutput
      activationOutput
      activationGeneratorOutput,
    HasForward
      (LMHeadLayerNormF style device dataType inputEmbedDim)
      activationOutput
      activationGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (LMHeadDecoderF style device dataType inputEmbedDim vocabDim)
      layerNormOutput
      layerNormGeneratorOutput
      decoderOutput
      generatorOutput,
    decoderOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    output ~ LMHeadOutputF style decoderOutput device dataType vocabDim
  ) =>
  HasForward
    (LMHead style device dataType inputEmbedDim vocabDim)
    input
    generator
    output
    generatorOutput
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
     in runIxState $
          ireturn input
            >>>= IxState . forward lmHeadDense
            >>>= IxState . forward lmHeadActivation
            >>>= IxState . forward lmHeadLayerNorm
            >>>= IxState . forward lmHeadDecoder
            >>>= ireturn . scaling (sing @style)
            >>>= ireturn . bias (sing @style)
