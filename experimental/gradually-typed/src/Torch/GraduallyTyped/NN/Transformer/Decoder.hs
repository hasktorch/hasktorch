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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Decoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack, TransformerDecoderStack, lookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), KnownDim, Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SelectDim (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

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
data
  TransformerDecoder
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
    forall numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerDecoderF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP ->
    TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP

type GTransformerDecoderF
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GTransformerDecoder
    (TDStackF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
    (TDEmbedLayerNormF style device dataType decoderInputEmbedDim)
    (TDLayerNormF style device dataType decoderInputEmbedDim)
    (TDDropoutF style dropoutP)
    (TDPosEncF style device dataType headDim decoderInputEmbedDim posEncDim)

type family
  TDStackF
    (numLayers :: Nat)
    (style :: TransformerStyle)
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
  TDStackF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP =
    TransformerDecoderStack numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP

type family
  TDEmbedLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDEmbedLayerNormF 'T5 _ _ _ = ()
  TDEmbedLayerNormF 'ByT5 device dataType decoderInputEmbedDim = TDEmbedLayerNormF 'T5 device dataType decoderInputEmbedDim
  TDEmbedLayerNormF 'BART device dataType decoderInputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[decoderInputEmbedDim])
  TDEmbedLayerNormF 'MBART device dataType decoderInputEmbedDim = TDEmbedLayerNormF 'BART device dataType decoderInputEmbedDim
  TDEmbedLayerNormF 'Pegasus _ _ _ = ()

type family
  TDLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDLayerNormF 'T5 device dataType decoderInputEmbedDim = LayerNorm 'WithoutBias device dataType ('Shape '[decoderInputEmbedDim])
  TDLayerNormF 'ByT5 device dataType decoderInputEmbedDim = TDLayerNormF 'T5 device dataType decoderInputEmbedDim
  TDLayerNormF 'BART _ _ _ = ()
  TDLayerNormF 'MBART device dataType decoderInputEmbedDim = TDLayerNormF 'BART device dataType decoderInputEmbedDim
  TDLayerNormF 'Pegasus device dataType decoderInputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[decoderInputEmbedDim])

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
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDPosEncF 'T5 device dataType headDim _ posEncDim = Embedding ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TDPosEncF 'ByT5 device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'T5 device dataType headDim decoderInputEmbedDim posEncDim
  TDPosEncF 'BART device dataType _ decoderInputEmbedDim posEncDim = Embedding ('Layout 'Dense) device dataType posEncDim decoderInputEmbedDim 'Nothing
  TDPosEncF 'MBART device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'BART device dataType headDim decoderInputEmbedDim posEncDim
  TDPosEncF 'Pegasus device dataType headDim decoderInputEmbedDim posEncDim = TDPosEncF 'BART device dataType headDim decoderInputEmbedDim posEncDim

instance
  ( SingI style,
    stack ~ TDStackF numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
    HasInitialize stack,
    embedLayerNorm ~ TDEmbedLayerNormF style device dataType decoderInputEmbedDim,
    HasInitialize embedLayerNorm,
    layerNorm ~ TDLayerNormF style device dataType decoderInputEmbedDim,
    HasInitialize layerNorm,
    dropout ~ TDDropoutF style dropoutP,
    HasInitialize dropout,
    posEnc ~ TDPosEncF style device dataType headDim decoderInputEmbedDim posEncDim,
    HasInitialize posEnc
  ) =>
  HasInitialize (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
  where
  type
    InitializeF (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim decoderInputEmbedDim ->
      SDim encoderOutputEmbedDim ->
      SDim ffnDim ->
      SDim posEncDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps =
    runState $ do
      stack <-
        state $ initialize @stack device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP eps
      let embedLayerNorm = case sing @style of
            ST5 -> ()
            SByT5 -> ()
            SBART -> initialize @embedLayerNorm device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
            SMBART -> initialize @embedLayerNorm device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
            SPegasus -> ()
            SBERT -> undefined
            SRoBERTa -> undefined
            SGPT2 -> undefined
      let layerNorm = case sing @style of
            ST5 -> initialize @layerNorm device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
            SByT5 -> initialize @layerNorm device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
            SBART -> ()
            SMBART -> ()
            SPegasus -> initialize @layerNorm device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
            SBERT -> undefined
            SRoBERTa -> undefined
            SGPT2 -> undefined
      let dropout = initialize @dropout dropoutP
      posEnc <-
        state $ case sing @style of
          ST5 -> initialize @posEnc (SLayout SDense) device dataType posEncDim headDim
          SByT5 -> initialize @posEnc (SLayout SDense) device dataType posEncDim headDim
          SBART -> initialize @posEnc (SLayout SDense) device dataType posEncDim decoderInputEmbedDim
          SMBART -> initialize @posEnc (SLayout SDense) device dataType posEncDim decoderInputEmbedDim
          SPegasus -> initialize @posEnc (SLayout SDense) device dataType posEncDim decoderInputEmbedDim
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
      pure . TransformerDecoder $ GTransformerDecoder stack embedLayerNorm layerNorm dropout posEnc

lookupDecoder ::
  forall numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim embedDim,
    KnownDim decoderInputEmbedDim,
    KnownDim encoderOutputEmbedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    Scalar dropoutP,
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP m
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  Double ->
  String ->
  m (TransformerDecoder numLayers style device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
lookupDecoder headDim headEmbedDim embedDim dropoutP eps prefix =
  let stack ST5 = lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps (prefix <> "block.")
      stack SByT5 = lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps (prefix <> "block.")
      stack SBART = lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SMBART = lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SPegasus = lookupDecoderStack headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SBERT = undefined
      stack SRoBERTa = undefined
      stack SGPT2 = undefined
      embedLayerNorm ST5 = pure ()
      embedLayerNorm SByT5 = pure ()
      embedLayerNorm SBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "layernorm_embedding.weight")
          <*> lookupTensor (prefix <> "layernorm_embedding.bias")
          <*> pure eps
      embedLayerNorm SMBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "layernorm_embedding.weight")
          <*> lookupTensor (prefix <> "layernorm_embedding.bias")
          <*> pure eps
      embedLayerNorm SPegasus = pure ()
      embedLayerNorm SBERT = undefined
      embedLayerNorm SRoBERTa = undefined
      embedLayerNorm SGPT2 = undefined
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> pure eps
      layerNorm SByT5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> pure eps
      layerNorm SBART = pure ()
      layerNorm SMBART = pure ()
      layerNorm SPegasus =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> lookupTensor (prefix <> "layer_norm.bias")
          <*> pure eps
      layerNorm SBERT = undefined
      layerNorm SRoBERTa = undefined
      layerNorm SGPT2 = undefined
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
      posEnc ST5 = Embedding <$> lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
      posEnc SByT5 = Embedding <$> lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
      posEnc SBART = Embedding <$> lookupTensor (prefix <> "embed_positions.weight")
      posEnc SMBART = Embedding <$> lookupTensor (prefix <> "embed_positions.weight")
      posEnc SPegasus = Embedding <$> lookupTensor (prefix <> "embed_positions.weight")
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
      (TransformerDecoderStack numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          'WithGradient
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
          crossAttentionMaskRequiresGradient
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
    (TransformerDecoder numLayers 'T5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderInput
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxState $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxState . forward tdLayerNorm
            >>>= IxState . forward tdDropout

testDecoder = do
  let device = SDevice SCPU
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
  let (encoder, g') = initialize @(TransformerDecoder 1 'T5 _ _ _ _ _ _ _ _ _ _) device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = sOnes SWithoutGradient (SLayout SDense) device
      decoderInput = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderInputEmbedDim :|: SNil)
      encoderOutput = sOnes' dataType (SShape $ batchDim :|: seqDim :|: encoderOutputEmbedDim :|: SNil)
      decoderRelPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: seqDim :|: SNil)
  let (output, _) = forward encoder (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) g'
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
      (TransformerDecoderStack numLayers 'ByT5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          'WithGradient
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
          crossAttentionMaskRequiresGradient
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
    (TransformerDecoder numLayers 'ByT5 device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderInput
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxState $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxState . forward tdLayerNorm
            >>>= IxState . forward tdDropout

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
      (TDEmbedLayerNormF 'BART device dataType decoderInputEmbedDim)
      ( Tensor
          'WithGradient
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
      (TDStackF numLayers 'BART device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          decoderAttentionMaskRequiresGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape),
        Tensor
          crossAttentionMaskRequiresGradient
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
    (TransformerDecoder numLayers 'BART device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosRequiresGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxState . forward tdEmbedLayerNorm
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     IxState $
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
          'WithGradient
          (decoderInputLayout <+> 'Layout 'Dense <+> decoderPosLayout)
          (decoderInputDevice <+> device <+> decoderPosDevice)
          (decoderInputDataType <+> Seq (decoderPosDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF decoderInputShape (EmbeddingF ('Shape '[posEncDim, decoderInputEmbedDim]) decoderPosShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TDStackF numLayers 'Pegasus device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          decoderAttentionMaskRequiresGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape),
        Tensor
          crossAttentionMaskRequiresGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) crossAttentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TDLayerNormF 'Pegasus device dataType decoderInputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder numLayers 'Pegasus device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      encoderOutput,
      Tensor decoderPosRequiresGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerDecoder GTransformerDecoder {..}) (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderPos
            >>>= IxState . forward tdPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     IxState $
                       forward
                         tdStack
                         ( decoderInput',
                           encoderOutput,
                           decoderAttentionBias,
                           crossAttentionBias
                         )
                 )
            >>>= IxState . forward tdLayerNorm