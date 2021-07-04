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
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType, SDataType)
import Torch.GraduallyTyped.Device (Device, DeviceType, SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sZeros)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
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

type family
  HasInitializeLMHeadDenseInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeLMHeadDenseInputF 'T5 _ _ _ _ _ = ()
  HasInitializeLMHeadDenseInputF 'ByT5 gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDenseInputF 'T5 gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDenseInputF 'BART _ _ _ _ _ = ()
  HasInitializeLMHeadDenseInputF 'MBART gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDenseInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDenseInputF 'Pegasus gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDenseInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDenseInputF 'BERT gradient device dataType inputEmbedDim _ = (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim inputEmbedDim)
  HasInitializeLMHeadDenseInputF 'RoBERTa gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDenseInputF 'BERT gradient device dataType inputEmbedDim vocabDim

type family
  HasInitializeLMHeadActivationInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeLMHeadActivationInputF 'T5 _ _ _ _ _ = ()
  HasInitializeLMHeadActivationInputF 'ByT5 gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadActivationInputF 'T5 gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadActivationInputF 'BART _ _ _ _ _ = ()
  HasInitializeLMHeadActivationInputF 'MBART gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadActivationInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadActivationInputF 'Pegasus gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadActivationInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadActivationInputF 'BERT _ _ _ _ _ = ()
  HasInitializeLMHeadActivationInputF 'RoBERTa gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadActivationInputF 'BERT gradient device dataType inputEmbedDim vocabDim

type family
  HasInitializeLMHeadLayerNormInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeLMHeadLayerNormInputF 'T5 _ _ _ _ _ = ()
  HasInitializeLMHeadLayerNormInputF 'ByT5 gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadLayerNormInputF 'T5 gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadLayerNormInputF 'BART _ _ _ _ _ = ()
  HasInitializeLMHeadLayerNormInputF 'MBART gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadLayerNormInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadLayerNormInputF 'Pegasus gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadLayerNormInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadLayerNormInputF 'BERT gradient device dataType inputEmbedDim _ = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeLMHeadLayerNormInputF 'RoBERTa gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadLayerNormInputF 'BERT gradient device dataType inputEmbedDim vocabDim

type family
  HasInitializeLMHeadDecoderInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeLMHeadDecoderInputF 'T5 gradient device dataType inputEmbedDim vocabDim = (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim)
  HasInitializeLMHeadDecoderInputF 'ByT5 gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDecoderInputF 'T5 gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDecoderInputF 'BART gradient device dataType inputEmbedDim vocabDim = (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim)
  HasInitializeLMHeadDecoderInputF 'MBART gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDecoderInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDecoderInputF 'Pegasus gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDecoderInputF 'BART gradient device dataType inputEmbedDim vocabDim
  HasInitializeLMHeadDecoderInputF 'BERT gradient device dataType inputEmbedDim vocabDim = (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim)
  HasInitializeLMHeadDecoderInputF 'RoBERTa gradient device dataType inputEmbedDim vocabDim = HasInitializeLMHeadDecoderInputF 'BERT gradient device dataType inputEmbedDim vocabDim

instance
  ( SingI style,
    dense ~ LMHeadDenseF style gradient device dataType inputEmbedDim,
    HasInitialize dense (HasInitializeLMHeadDenseInputF style gradient device dataType inputEmbedDim vocabDim) generator generator',
    activation ~ LMHeadActivationF style,
    HasInitialize activation (HasInitializeLMHeadActivationInputF style gradient device dataType inputEmbedDim vocabDim) generator' generator'',
    layerNorm ~ LMHeadLayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize layerNorm (HasInitializeLMHeadLayerNormInputF style gradient device dataType inputEmbedDim vocabDim) generator'' generator''',
    decoder ~ LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim,
    HasInitialize decoder (HasInitializeLMHeadDecoderInputF style gradient device dataType inputEmbedDim vocabDim) generator''' generator'''',
    bias ~ LMHeadBiasF style gradient device dataType vocabDim
  ) =>
  HasInitialize
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
    (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim, Double)
    generator
    generator''''
  where
  initialize (gradient, device, dataType, inputEmbedDim, vocabDim, eps) =
    let dense = IxState . initialize $
          case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> (gradient, device, dataType, inputEmbedDim, inputEmbedDim)
            SRoBERTa -> (gradient, device, dataType, inputEmbedDim, inputEmbedDim)
            SGPT2 -> undefined
        activation = IxState . initialize $
          case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        layerNorm = IxState . initialize $
          case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
            SRoBERTa -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
            SGPT2 -> undefined
        decoder = IxState . initialize $
          case sing @style of
            ST5 -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SByT5 -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SBART -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SMBART -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SPegasus -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SBERT -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SRoBERTa -> (gradient, device, dataType, inputEmbedDim, vocabDim)
            SGPT2 -> undefined
        bias = ireturn $
          case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> sZeros gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SMBART -> sZeros gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SPegasus -> sZeros gradient (SLayout SDense) device dataType (SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil)
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
     in runIxState $
          (GLMHead <<$>> ireturn inputEmbedDim <<*>> dense <<*>> activation <<*>> layerNorm <<*>> decoder <<*>> bias)
            >>>= ireturn . LMHead

instance
  SingI style =>
  HasStateDict
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
    (SGradient gradient, SDevice device, SDataType dataType, SDim inputEmbedDim, SDim vocabDim, Double)
  where
  fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim, eps) k =
    let dense ST5 = pure ()
        dense SByT5 = pure ()
        dense SBART = pure ()
        dense SMBART = pure ()
        dense SPegasus = pure ()
        dense SBERT = fromStateDict (gradient, device, dataType, inputEmbedDim, inputEmbedDim) (k <> "transform.dense.")
        dense SRoBERTa = fromStateDict (gradient, device, dataType, inputEmbedDim, inputEmbedDim) (k <> "dense.")
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
        layerNorm SBERT = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "transform.LayerNorm.")
        layerNorm SRoBERTa = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SGPT2 = undefined
        decoder ST5 = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) k
        decoder SByT5 = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) k
        decoder SBART = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) (k <> "lm_head.")
        decoder SMBART = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) (k <> "lm_head.")
        decoder SPegasus = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) (k <> "lm_head.")
        decoder SBERT = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) (k <> "decoder.")
        decoder SRoBERTa = fromStateDict (gradient, device, dataType, inputEmbedDim, vocabDim) (k <> "decoder.")
        decoder SGPT2 = undefined
        bias ST5 = pure ()
        bias SByT5 = pure ()
        bias SBART = fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil) (k <> "final_logits_bias")
        bias SMBART = fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil) (k <> "final_logits_bias")
        bias SPegasus = fromStateDict (gradient, SLayout SDense, device, dataType, SShape $ SName @"*" :&: SSize @1 :|: vocabDim :|: SNil) (k <> "final_logits_bias")
        bias SBERT = pure ()
        bias SRoBERTa = pure ()
        bias SGPT2 = undefined
     in LMHead
          <$> ( GLMHead inputEmbedDim
                  <$> dense (sing @style)
                  <*> pure (activation $ sing @style)
                  <*> layerNorm (sing @style)
                  <*> decoder (sing @style)
                  <*> bias (sing @style)
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
      (LMHeadLayerNormF style gradient device dataType inputEmbedDim)
      activationOutput
      activationGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (LMHeadDecoderF style gradient device dataType inputEmbedDim vocabDim)
      layerNormOutput
      layerNormGeneratorOutput
      decoderOutput
      generatorOutput,
    decoderOutput ~ Tensor gradient' layout' device' dataType' shape',
    output ~ LMHeadOutputF style decoderOutput gradient device dataType vocabDim
  ) =>
  HasForward
    (LMHead style gradient device dataType inputEmbedDim vocabDim)
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
