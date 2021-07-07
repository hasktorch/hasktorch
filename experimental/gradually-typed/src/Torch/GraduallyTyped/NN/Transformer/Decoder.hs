{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrRightAssociativeL #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Decoder where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (TransformerDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SelectDim (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic transformer decoder.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerDecoder'.
data
  GTransformerDecoder
    (stack :: Type)
    (embedLayerNorm :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
    (posEnc :: Type)
  where
  GTransformerDecoder ::
    forall stack embedLayerNorm layerNorm dropout posEnc.
    { -- | decoder layer stack
      tdStack :: stack,
      -- | decoder embedding layer norm
      tdEmbedLayerNorm :: embedLayerNorm,
      -- | decoder layer norm
      tdLayerNorm :: layerNorm,
      -- | decoder dropout
      tdDropout :: dropout,
      -- | positional encoding
      tdPosEnc :: posEnc
    } ->
    GTransformerDecoder stack embedLayerNorm layerNorm dropout posEnc

-- | Transformer decoder.
newtype
  TransformerDecoder
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoder ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerDecoder
      (TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      (TDEmbedLayerNormF style gradient device dataType decoderInputEmbedDim)
      (TDLayerNormF style gradient device dataType decoderInputEmbedDim)
      (TDDropoutF style dropoutP)
      (TDPosEncF style gradient device dataType headDim decoderInputEmbedDim posEncDim) ->
    TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP

type family
  TDStackF
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP =
    TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP

type family
  TDEmbedLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDEmbedLayerNormF 'T5 _ _ _ _ = ()
  TDEmbedLayerNormF 'ByT5 gradient device dataType decoderInputEmbedDim = TDEmbedLayerNormF 'T5 gradient device dataType decoderInputEmbedDim
  TDEmbedLayerNormF 'BART gradient device dataType decoderInputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[decoderInputEmbedDim])
  TDEmbedLayerNormF 'MBART gradient device dataType decoderInputEmbedDim = TDEmbedLayerNormF 'BART gradient device dataType decoderInputEmbedDim
  TDEmbedLayerNormF 'Pegasus _ _ _ _ = ()

type family
  TDLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDLayerNormF 'T5 gradient device dataType decoderInputEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[decoderInputEmbedDim])
  TDLayerNormF 'ByT5 gradient device dataType decoderInputEmbedDim = TDLayerNormF 'T5 gradient device dataType decoderInputEmbedDim
  TDLayerNormF 'BART _ _ _ _ = ()
  TDLayerNormF 'MBART gradient device dataType decoderInputEmbedDim = TDLayerNormF 'BART gradient device dataType decoderInputEmbedDim
  TDLayerNormF 'Pegasus gradient device dataType decoderInputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[decoderInputEmbedDim])

type family
  TDDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  TDDropoutF _ dropoutP = Dropout dropoutP

type family
  TDPosEncF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDPosEncF 'T5 gradient device dataType headDim _ posEncDim = Embedding gradient ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TDPosEncF 'ByT5 gradient device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'T5 gradient device dataType headDim decoderInputEmbedDim posEncDim
  TDPosEncF 'BART gradient device dataType _ decoderInputEmbedDim posEncDim = Embedding gradient ('Layout 'Dense) device dataType posEncDim decoderInputEmbedDim 'Nothing
  TDPosEncF 'MBART gradient device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'BART gradient device dataType headDim decoderInputEmbedDim posEncDim
  TDPosEncF 'Pegasus gradient device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'BART gradient device dataType headDim decoderInputEmbedDim posEncDim

type family
  HasInitializeTDEmbedLayerNormInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTDEmbedLayerNormInputF 'T5 _ _ _ _ = ()
  HasInitializeTDEmbedLayerNormInputF 'ByT5 gradient device dataType decoderInputEmbedDim = HasInitializeTDEmbedLayerNormInputF 'T5 gradient device dataType decoderInputEmbedDim
  HasInitializeTDEmbedLayerNormInputF 'BART gradient device dataType decoderInputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[decoderInputEmbedDim]), Double)
  HasInitializeTDEmbedLayerNormInputF 'MBART gradient device dataType decoderInputEmbedDim = HasInitializeTDEmbedLayerNormInputF 'BART gradient device dataType decoderInputEmbedDim
  HasInitializeTDEmbedLayerNormInputF 'Pegasus _ _ _ _ = ()

type family
  HasInitializeTDLayerNormInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTDLayerNormInputF 'T5 gradient device dataType decoderInputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[decoderInputEmbedDim]), Double)
  HasInitializeTDLayerNormInputF 'ByT5 gradient device dataType decoderInputEmbedDim = HasInitializeTDLayerNormInputF 'T5 gradient device dataType decoderInputEmbedDim
  HasInitializeTDLayerNormInputF 'BART _ _ _ _ = ()
  HasInitializeTDLayerNormInputF 'MBART gradient device dataType decoderInputEmbedDim = HasInitializeTDLayerNormInputF 'BART gradient device dataType decoderInputEmbedDim
  HasInitializeTDLayerNormInputF 'Pegasus gradient device dataType decoderInputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[decoderInputEmbedDim]), Double)

type family
  HasInitializeTDPosEncInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTDPosEncInputF 'T5 gradient device dataType headDim _ posEncDim = (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim headDim)
  HasInitializeTDPosEncInputF 'ByT5 gradient device dataType headDim decoderInputEmbedDim posEncDim = HasInitializeTDPosEncInputF 'T5 gradient device dataType headDim decoderInputEmbedDim posEncDim
  HasInitializeTDPosEncInputF 'BART gradient device dataType _ decoderInputEmbedDim posEncDim = (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim decoderInputEmbedDim)
  HasInitializeTDPosEncInputF 'MBART gradient device dataType headDim decoderInputEmbedDim posEncDim = HasInitializeTDPosEncInputF 'BART gradient device dataType headDim decoderInputEmbedDim posEncDim
  HasInitializeTDPosEncInputF 'Pegasus gradient device dataType headDim decoderInputEmbedDim posEncDim = HasInitializeTDPosEncInputF 'BART gradient device dataType headDim decoderInputEmbedDim posEncDim

instance
  ( SingI style,
    stack ~ TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
    HasInitialize stack (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim decoderInputEmbedDim, SDim encoderOutputEmbedDim, SDim ffnDim, dropoutP, Double) generator generator',
    embedLayerNorm ~ TDEmbedLayerNormF style gradient device dataType decoderInputEmbedDim,
    HasInitialize embedLayerNorm (HasInitializeTDEmbedLayerNormInputF style gradient device dataType decoderInputEmbedDim) generator' generator'',
    layerNorm ~ TDLayerNormF style gradient device dataType decoderInputEmbedDim,
    HasInitialize layerNorm (HasInitializeTDLayerNormInputF style gradient device dataType decoderInputEmbedDim) generator'' generator''',
    dropout ~ TDDropoutF style dropoutP,
    HasInitialize dropout dropoutP generator''' generator''',
    posEnc ~ TDPosEncF style gradient device dataType headDim decoderInputEmbedDim posEncDim,
    HasInitialize posEnc (HasInitializeTDPosEncInputF style gradient device dataType headDim decoderInputEmbedDim posEncDim) generator''' generator''''
  ) =>
  HasInitialize
    (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim decoderInputEmbedDim, SDim encoderOutputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
    generator
    generator''''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, posEncDim, dropoutP, eps) =
    let decoderStack = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps)
        embedLayerNorm = IxState . initialize $ case sing @style of
          ST5 -> ()
          SByT5 -> ()
          SBART -> (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps)
          SMBART -> (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps)
          SPegasus -> ()
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        layerNorm = IxState . initialize $ case sing @style of
          ST5 -> (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps)
          SByT5 -> (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps)
          SBART -> ()
          SMBART -> ()
          SPegasus -> (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps)
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        dropout = IxState . initialize $ dropoutP
        posEnc = IxState . initialize $ case sing @style of
          ST5 -> (gradient, SLayout SDense, device, dataType, posEncDim, headDim)
          SByT5 -> (gradient, SLayout SDense, device, dataType, posEncDim, headDim)
          SBART -> (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim)
          SMBART -> (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim)
          SPegasus -> (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim)
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
     in runIxState $
          (GTransformerDecoder <<$>> decoderStack <<*>> embedLayerNorm <<*>> layerNorm <<*>> dropout <<*>> posEnc) >>>= ireturn . TransformerDecoder

instance
  (SingI style, KnownNat numLayers) =>
  HasStateDict
    (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim decoderInputEmbedDim, SDim encoderOutputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, posEncDim, dropoutP, eps) k =
    let stack ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps) (k <> "block.")
        stack SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps) (k <> "block.")
        stack SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SBERT = undefined
        stack SRoBERTa = undefined
        stack SGPT2 = undefined
        embedLayerNorm ST5 = fromStateDict () k
        embedLayerNorm SByT5 = fromStateDict () k
        embedLayerNorm SBART = fromStateDict (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps) (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = fromStateDict (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps) (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = fromStateDict () k
        embedLayerNorm SBERT = undefined
        embedLayerNorm SRoBERTa = undefined
        embedLayerNorm SGPT2 = undefined
        layerNorm ST5 = fromStateDict (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SByT5 = fromStateDict (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SBART = fromStateDict () k
        layerNorm SMBART = fromStateDict () k
        layerNorm SPegasus = fromStateDict (gradient, device, dataType, SShape $ decoderInputEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
        posEnc ST5 = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, headDim) (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, headDim) (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim) (k <> "embed_positions.")
        posEnc SMBART = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim) (k <> "embed_positions.")
        posEnc SPegasus = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, decoderInputEmbedDim) (k <> "embed_positions.")
        posEnc SBERT = undefined
        posEnc SRoBERTa = undefined
        posEnc SGPT2 = undefined
     in TransformerDecoder
          <$> ( GTransformerDecoder
                  <$> stack (sing @style)
                  <*> embedLayerNorm (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
                  <*> posEnc (sing @style)
              )
  toStateDict k (TransformerDecoder GTransformerDecoder {..}) =
    let stack ST5 = toStateDict (k <> "block.")
        stack SByT5 = toStateDict (k <> "block.")
        stack SBART = toStateDict (k <> "layers.")
        stack SMBART = toStateDict (k <> "layers.")
        stack SPegasus = toStateDict (k <> "layers.")
        stack SBERT = undefined
        stack SRoBERTa = undefined
        stack SGPT2 = undefined
        embedLayerNorm ST5 = toStateDict k
        embedLayerNorm SByT5 = toStateDict k
        embedLayerNorm SBART = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = toStateDict k
        embedLayerNorm SBERT = undefined
        embedLayerNorm SRoBERTa = undefined
        embedLayerNorm SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "final_layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "final_layer_norm.")
        layerNorm SBART = toStateDict k
        layerNorm SMBART = toStateDict k
        layerNorm SPegasus = toStateDict (k <> "layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = toStateDict k
        posEnc ST5 = toStateDict (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = toStateDict (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = toStateDict (k <> "embed_positions.")
        posEnc SMBART = toStateDict (k <> "embed_positions.")
        posEnc SPegasus = toStateDict (k <> "embed_positions.")
        posEnc SBERT = undefined
        posEnc SRoBERTa = undefined
        posEnc SGPT2 = undefined
     in do
          () <- stack (sing @style) tdStack
          () <- embedLayerNorm (sing @style) tdEmbedLayerNorm
          () <- layerNorm (sing @style) tdLayerNorm
          () <- dropout (sing @style) tdDropout
          () <- posEnc (sing @style) tdPosEnc
          pure ()

-- | 'HasForward' instance for @TransformerDecoder numLayers 'T5@.
--
-- @
-- ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ encoderOutput │  │ decoderRelPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  └─────────┬──────────┘
--        │                  │                  │                     │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              tdPosEnc                  │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              transpose                 │                        │
--        │                  │                  ▼                     ▼                        ▼
--        │                  │              transpose             unsqueeze                unsqueeze
--        ▼                  │                  │                     │                        │
--    tdDropout              │                  └────────►add◄────────┘                        │
--        ▼                  │                             │                                   │
--     tdStack◄──────────────┘◄────────────────────────────┘◄──────────────────────────────────┘
--        ▼
--   tdLayerNorm
--        ▼
--    tdDropout
--        │
--        ▼
--    ┌────────┐
--    │ output │
--    └────────┘
-- @
instance
  ( HasForward
      (Dropout dropoutP)
      decoderInput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TransformerDecoderStack 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          (gradient <|> decoderRelPosGradient <|> decoderAttentionMaskGradient)
          ('Layout 'Dense <+> decoderRelPosLayout <+> decoderAttentionMaskLayout)
          (device <+> decoderRelPosDevice <+> decoderAttentionMaskDevice)
          (Seq (decoderRelPosDataType <+> 'DataType 'Int64) dataType <+> decoderAttentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          decoderRelPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  decoderAttentionMaskShape
              )
          ),
        Tensor
          crossAttentionMaskGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          ( UnsqueezeF
              ('SelectDim ('ByIndex 1))
              crossAttentionMaskShape
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      ( LayerNorm
          'WithoutBias
          gradient
          device
          dataType
          ('Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (Dropout dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxStateT . forward tdPosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderInput
            >>>= IxStateT . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxStateT $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxStateT . forward tdLayerNorm
            >>>= IxStateT . forward tdDropout

testDecoder = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      decoderInputEmbedDim = SName @"*" :&: SSize @512
      encoderOutputEmbedDim = decoderInputEmbedDim
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (encoder, g') = initialize @(TransformerDecoder 'T5 1 _ _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, decoderInputEmbedDim, encoderOutputEmbedDim, ffnDim, posEncDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      decoderInput = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderInputEmbedDim :|: SNil)
      encoderOutput = sOnes' dataType (SShape $ batchDim :|: seqDim :|: encoderOutputEmbedDim :|: SNil)
      decoderRelPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoder (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) g'
  pure output

-- | 'HasForward' instance for @TransformerDecoder numLayers 'ByT5@.
--
-- @
-- ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ encoderOutput │  │ decoderRelPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  └─────────┬──────────┘
--        │                  │                  │                     │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              tdPosEnc                  │                        │
--        │                  │                  ▼                     │                        │
--        │                  │              transpose                 │                        │
--        │                  │                  ▼                     ▼                        ▼
--        │                  │              transpose             unsqueeze                unsqueeze
--        ▼                  │                  │                     │                        │
--    tdDropout              │                  └────────►add◄────────┘                        │
--        ▼                  │                             │                                   │
--     tdStack◄──────────────┘◄────────────────────────────┘◄──────────────────────────────────┘
--        ▼
--   tdLayerNorm
--        ▼
--    tdDropout
--        │
--        ▼
--    ┌────────┐
--    │ output │
--    └────────┘
-- @
instance
  ( HasForward
      (Dropout dropoutP)
      decoderInput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TransformerDecoderStack 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          (gradient <|> decoderRelPosGradient <|> decoderAttentionMaskGradient)
          ('Layout 'Dense <+> decoderRelPosLayout <+> decoderAttentionMaskLayout)
          (device <+> decoderRelPosDevice <+> decoderAttentionMaskDevice)
          (Seq (decoderRelPosDataType <+> 'DataType 'Int64) dataType <+> decoderAttentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          decoderRelPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  decoderAttentionMaskShape
              )
          ),
        Tensor
          crossAttentionMaskGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          ( UnsqueezeF
              ('SelectDim ('ByIndex 1))
              crossAttentionMaskShape
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      ( LayerNorm
          'WithoutBias
          gradient
          device
          dataType
          ('Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (Dropout dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxStateT . forward tdPosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderInput
            >>>= IxStateT . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxStateT $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxStateT . forward tdLayerNorm
            >>>= IxStateT . forward tdDropout

-- | 'HasForward' instance for @TransformerDecoder numLayers 'BART@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     │                         │
--          tdEmbedLayerNorm                 │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @
instance
  ( HasForward
      (TDEmbedLayerNormF 'BART gradient device dataType decoderInputEmbedDim)
      ( Tensor
          (decoderInputGradient <|> gradient <|> decoderPosGradient)
          (decoderInputLayout <+> 'Layout 'Dense <+> decoderPosLayout)
          (decoderInputDevice <+> device <+> decoderPosDevice)
          (decoderInputDataType <+> Seq (decoderPosDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF decoderInputShape (EmbeddingF ('Shape '[posEncDim, decoderInputEmbedDim]) decoderPosShape))
      )
      generator
      layerNormOutput
      generator,
    HasForward
      (TDDropoutF 'BART dropoutP)
      layerNormOutput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TDStackF 'BART numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          decoderAttentionMaskGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape),
        Tensor
          crossAttentionMaskGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) crossAttentionMaskShape)
      )
      dropoutGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder 'BART numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderPos
            >>>= IxStateT . forward tdPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxStateT . forward tdEmbedLayerNorm
            >>>= IxStateT . forward tdDropout
            >>>= ( \decoderInput' ->
                     IxStateT $
                       forward
                         tdStack
                         ( decoderInput',
                           encoderOutput,
                           decoderAttentionBias,
                           crossAttentionBias
                         )
                 )

-- | 'HasForward' instance for @TransformerDecoder numLayers 'MBART@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     │                         │
--          tdEmbedLayerNorm                 │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 ▼
--            tdLayerNorm
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @

-- | 'HasForward' instance for @TransformerDecoder numLayers 'Pegasus@.
--
-- @
-- ┌──────────────┐  ┌────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ decoderInput │  │ decoderPos │  │ encoderOutput │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └──────┬───────┘  └──────┬─────┘  └───────┬───────┘  └──────────┬───────────┘  └──────────┬─────────┘
--        │                 │                │                     │                         │
--        │                 ▼                │                     │                         │
--        │             tdPosEnc             │                     │                         │
--        │                 │                │                     │                         │
--        └──────►add◄──────┘                │                     │                         │
--                 │                         │                     │                         │
--                 ▼                         │                     ▼                         ▼
--             tdDropout                     │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tdStack◄─────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 ▼
--            tdLayerNorm
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @
instance
  ( HasForward
      (TDDropoutF 'Pegasus dropoutP)
      ( Tensor
          (decoderInputGradient <|> gradient <|> decoderPosGradient)
          (decoderInputLayout <+> 'Layout 'Dense <+> decoderPosLayout)
          (decoderInputDevice <+> device <+> decoderPosDevice)
          (decoderInputDataType <+> Seq (decoderPosDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF decoderInputShape (EmbeddingF ('Shape '[posEncDim, decoderInputEmbedDim]) decoderPosShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TDStackF 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          decoderAttentionMaskGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape),
        Tensor
          crossAttentionMaskGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) crossAttentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TDLayerNormF 'Pegasus gradient device dataType decoderInputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderPos
            >>>= IxStateT . forward tdPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxStateT . forward tdDropout
            >>>= ( \decoderInput' ->
                     IxStateT $
                       forward
                         tdStack
                         ( decoderInput',
                           encoderOutput,
                           decoderAttentionBias,
                           crossAttentionBias
                         )
                 )
            >>>= IxStateT . forward tdLayerNorm