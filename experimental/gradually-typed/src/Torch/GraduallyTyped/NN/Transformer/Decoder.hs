{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
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
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat (SNat))
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (TransformerDecoderStack, TransformerDecoderStackSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (SWithBias, SWithoutBias))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SelectDim (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
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
  where
  TransformerDecoder ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim.
    GTransformerDecoder
      (TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
      (TDEmbedLayerNormF style gradient device dataType decoderInputEmbedDim)
      (TDLayerNormF style gradient device dataType decoderInputEmbedDim)
      (TDDropoutF style)
      (TDPosEncF style gradient device dataType headDim decoderInputEmbedDim posEncDim) ->
    TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim

data
  TransformerDecoderSpec
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
  where
  TransformerDecoderSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim.
    STransformerStyle style ->
    SNat numLayers ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim decoderInputEmbedDim ->
    SDim encoderOutputEmbedDim ->
    SDim ffnDim ->
    SDim posEncDim ->
    Double ->
    Double ->
    TransformerDecoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim

type instance ModelSpec (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim) = TransformerDecoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim

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
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim =
    TransformerDecoderStack style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim

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
    (style :: TransformerStyle) ::
    Type
  where
  TDDropoutF _ = Dropout

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

instance
  ( SingI style,
    decoderStack ~ TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim,
    HasInitialize decoderStack device decoderStack device,
    embedLayerNorm ~ TDEmbedLayerNormF style gradient device dataType decoderInputEmbedDim,
    HasInitialize embedLayerNorm device embedLayerNorm device,
    layerNorm ~ TDLayerNormF style gradient device dataType decoderInputEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    dropout ~ TDDropoutF style,
    HasInitialize dropout device dropout device,
    posEnc ~ TDPosEncF style gradient device dataType headDim decoderInputEmbedDim posEncDim,
    HasInitialize posEnc device posEnc device
  ) =>
  HasInitialize
    (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    generatorDevice
    (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    device
  where
  initialize (TransformerDecoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        decoderStack = IxStateT . initialize @decoderStack $ TransformerDecoderStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP eps
        embedLayerNormSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        embedLayerNorm = IxStateT . initialize @embedLayerNorm $ case sing @style of
          ST5 -> ()
          SByT5 -> ()
          SBART -> embedLayerNormSpec
          SMBART -> embedLayerNormSpec
          SPegasus -> ()
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $ case sing @style of
          ST5 -> layerNormWithoutBiasSpec
          SByT5 -> layerNormWithoutBiasSpec
          SBART -> ()
          SMBART -> ()
          SPegasus -> layerNormWithBiasSpec
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        dropout = IxStateT . initialize @dropout $ Dropout dropoutP
        relPosEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
        posEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim decoderInputEmbedDim SNothing
        posEnc = IxStateT . initialize @posEnc $ case sing @style of
          ST5 -> relPosEncSpec
          SByT5 -> relPosEncSpec
          SBART -> posEncSpec
          SMBART -> posEncSpec
          SPegasus -> posEncSpec
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        gtd =
          GTransformerDecoder
            <<$>> decoderStack
            <<*>> embedLayerNorm
            <<*>> layerNorm
            <<*>> dropout
            <<*>> posEnc
     in runIxStateT (gtd >>>= ireturn . TransformerDecoder) generator'

instance
  SingI style =>
  HasStateDict
    (TransformerDecoder style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
  where
  fromStateDict (TransformerDecoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps) k =
    let decoderStackSpec = TransformerDecoderStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP eps
        decoderStack ST5 = fromStateDict decoderStackSpec (k <> "block.")
        decoderStack SByT5 = fromStateDict decoderStackSpec (k <> "block.")
        decoderStack SBART = fromStateDict decoderStackSpec (k <> "layers.")
        decoderStack SMBART = fromStateDict decoderStackSpec (k <> "layers.")
        decoderStack SPegasus = fromStateDict decoderStackSpec (k <> "layers.")
        decoderStack SBERT = undefined
        decoderStack SRoBERTa = undefined
        decoderStack SGPT2 = undefined
        embedLayerNormSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        embedLayerNorm ST5 = fromStateDict () k
        embedLayerNorm SByT5 = fromStateDict () k
        embedLayerNorm SBART = fromStateDict embedLayerNormSpec (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = fromStateDict embedLayerNormSpec (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = fromStateDict () k
        embedLayerNorm SBERT = undefined
        embedLayerNorm SRoBERTa = undefined
        embedLayerNorm SGPT2 = undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
        layerNorm ST5 = fromStateDict layerNormWithoutBiasSpec (k <> "final_layer_norm.")
        layerNorm SByT5 = fromStateDict layerNormWithoutBiasSpec (k <> "final_layer_norm.")
        layerNorm SBART = fromStateDict () k
        layerNorm SMBART = fromStateDict () k
        layerNorm SPegasus = fromStateDict layerNormWithBiasSpec (k <> "layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict (Dropout dropoutP) k
        relPosEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
        posEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim decoderInputEmbedDim SNothing
        posEnc ST5 = fromStateDict relPosEncSpec (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = fromStateDict relPosEncSpec (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SMBART = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SPegasus = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SBERT = undefined
        posEnc SRoBERTa = undefined
        posEnc SGPT2 = undefined
     in TransformerDecoder
          <$> ( GTransformerDecoder
                  <$> decoderStack style
                  <*> embedLayerNorm style
                  <*> layerNorm style
                  <*> dropout style
                  <*> posEnc style
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
      Dropout
      decoderInput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TransformerDecoderStack 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      stackOutput
      stackGeneratorOutputDevice,
    HasForward
      ( LayerNorm
          'WithoutBias
          gradient
          device
          dataType
          ('Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutputDevice
      layerNormOutput
      layerNormGeneratorOutputDevice,
    HasForward
      Dropout
      layerNormOutput
      layerNormGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoder 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
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

testDecoder :: IO _
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
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (encoder, g') <- initialize (TransformerDecoderSpec ST5 (SNat @1) gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
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
      Dropout
      decoderInput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TransformerDecoderStack 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      stackOutput
      stackGeneratorOutputDevice,
    HasForward
      ( LayerNorm
          'WithoutBias
          gradient
          device
          dataType
          ('Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutputDevice
      layerNormOutput
      layerNormGeneratorOutputDevice,
    HasForward
      Dropout
      layerNormOutput
      layerNormGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoder 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
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
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (TDDropoutF 'BART)
      layerNormOutput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TDStackF 'BART numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoder 'BART numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
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
      (TDDropoutF 'Pegasus)
      ( Tensor
          (decoderInputGradient <|> gradient <|> decoderPosGradient)
          (decoderInputLayout <+> 'Layout 'Dense <+> decoderPosLayout)
          (decoderInputDevice <+> device <+> decoderPosDevice)
          (decoderInputDataType <+> Seq (decoderPosDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF decoderInputShape (EmbeddingF ('Shape '[posEncDim, decoderInputEmbedDim]) decoderPosShape))
      )
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TDStackF 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      stackOutput
      generatorOutputDevice,
    HasForward
      (TDLayerNormF 'Pegasus gradient device dataType decoderInputEmbedDim)
      stackOutput
      generatorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerDecoder 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
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