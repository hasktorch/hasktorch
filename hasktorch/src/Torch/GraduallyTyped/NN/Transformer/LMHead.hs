{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType)
import Torch.GraduallyTyped.DType (DataType)
import Torch.GraduallyTyped.Device (Device, DeviceType)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Gelu)
import Torch.GraduallyTyped.NN.Class (HasForward (..))
import Torch.GraduallyTyped.NN.Linear (Linear)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  GLMHead
    (dense :: Type)
    (activation :: Type)
    (layerNorm :: Type)
    (decoder :: Type)
    (bias :: Type)
  where
  GLMHead ::
    forall dense activation layerNorm decoder bias.
    { lmHeadDense :: dense,
      lmHeadActivation :: activation,
      lmHeadLayerNorm :: layerNorm,
      lmHeadDecoder :: decoder,
      lmHeadBias :: bias,
      lmHeadInputEmbedDim :: Dim String Integer
    } ->
    GLMHead dense activation layerNorm decoder bias

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
  LMHeadDenseF 'RoBERTa device dataType inputEmbedDim = Linear 'WithBias device dataType inputEmbedDim inputEmbedDim
  LMHeadDenseF 'T5 _ _ _ = ()
  LMHeadDenseF 'Pegasus _ _ _ = ()

type family
  LMHeadActivationF
    (style :: TransformerStyle) ::
    Type
  where
  LMHeadActivationF 'RoBERTa = Gelu
  LMHeadActivationF 'T5 = ()
  LMHeadActivationF 'Pegasus = ()

type family
  LMHeadLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadLayerNormF 'RoBERTa device dataType inputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  LMHeadLayerNormF 'T5 _ _ _ = ()
  LMHeadLayerNormF 'Pegasus _ _ _ = ()

type family
  LMHeadDecoderF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadDecoderF 'RoBERTa device dataType inputEmbedDim vocabDim = Linear 'WithBias device dataType vocabDim inputEmbedDim
  LMHeadDecoderF 'T5 device dataType inputEmbedDim vocabDim = Linear 'WithoutBias device dataType inputEmbedDim vocabDim
  LMHeadDecoderF 'Pegasus device dataType inputEmbedDim vocabDim = Linear 'WithoutBias device dataType inputEmbedDim vocabDim

type family
  LMHeadBiasF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  LMHeadBiasF 'RoBERTa _ _ _ = ()
  LMHeadBiasF 'Pegasus device dataType vocabDim = Tensor 'WithGradient ('Layout 'Dense) device dataType ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim])
  LMHeadBiasF 'T5 _ _ _ = ()

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
  LMHeadOutputF 'Pegasus (Tensor requiresGradient' layout' device' dataType' shape') device dataType vocabDim =
    Tensor
      'WithGradient
      (layout' <+> 'Layout 'Dense)
      (device' <+> device)
      (dataType' <+> dataType)
      (BroadcastShapesF shape' ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim]))
  LMHeadOutputF 'RoBERTa decoderOutput _ _ _ = decoderOutput

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
    let s :: Double = sqrt . fromIntegral . dimSize $ lmHeadInputEmbedDim
        scaling :: STransformerStyle style -> decoderOutput -> decoderOutput
        scaling ST5 = flip divScalar s
        scaling SPegasus = id
        scaling SRoBERTa = id
        bias :: STransformerStyle style -> decoderOutput -> output
        bias ST5 = id
        bias SPegasus = (`add` lmHeadBias)
        bias SRoBERTa = id
     in runIxState $
          ireturn input
            >>>= IxState . forward lmHeadDense
            >>>= IxState . forward lmHeadActivation
            >>>= IxState . forward lmHeadLayerNorm
            >>>= IxState . forward lmHeadDecoder
            >>>= ireturn . scaling (sing @style)
            >>>= ireturn . bias (sing @style)
