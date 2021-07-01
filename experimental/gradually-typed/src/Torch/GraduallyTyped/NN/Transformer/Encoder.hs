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

module Torch.GraduallyTyped.NN.Transformer.Encoder where

import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim, Name (..), SDim, SName (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Generic transformer encoder.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerEncoder'.
data
  GTransformerEncoder
    (stack :: Type)
    (embedLayerNorm :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
    (posEnc :: Type)
  where
  GTransformerEncoder ::
    forall stack embedLayerNorm layerNorm dropout posEnc.
    { -- | encoder layer stack
      teStack :: stack,
      -- | encoder embedding layer norm
      teEmbedLayerNorm :: embedLayerNorm,
      -- | encoder layer norm
      teLayerNorm :: layerNorm,
      -- | encoder dropout
      teDropout :: dropout,
      -- | positional encoding
      tePosEnc :: posEnc
    } ->
    GTransformerEncoder stack embedLayerNorm layerNorm dropout posEnc

-- | Transformer encoder.
newtype
  TransformerEncoder
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerEncoder ::
    forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerEncoderF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP ->
    TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type GTransformerEncoderF
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GTransformerEncoder
    (TEStackF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
    (TEEmbedLayerNormF style device dataType inputEmbedDim)
    (TELayerNormF style device dataType inputEmbedDim)
    (TEDropoutF style dropoutP)
    (TEPosEncF style device dataType headDim inputEmbedDim posEncDim)

type family
  TEStackF
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  TEStackF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP =
    TransformerStack numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP

type family
  TEEmbedLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEEmbedLayerNormF 'T5 _ _ _ = ()
  TEEmbedLayerNormF 'ByT5 device dataType inputEmbedDim = TEEmbedLayerNormF 'T5 device dataType inputEmbedDim
  TEEmbedLayerNormF 'BART device dataType inputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  TEEmbedLayerNormF 'MBART device dataType inputEmbedDim = TEEmbedLayerNormF 'BART device dataType inputEmbedDim
  TEEmbedLayerNormF 'Pegasus _ _ _ = ()
  TEEmbedLayerNormF 'BERT device dataType inputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  TEEmbedLayerNormF 'RoBERTa device dataType inputEmbedDim = TEEmbedLayerNormF 'BERT device dataType inputEmbedDim

type family
  TELayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TELayerNormF 'T5 device dataType inputEmbedDim = LayerNorm 'WithoutBias device dataType ('Shape '[inputEmbedDim])
  TELayerNormF 'ByT5 device dataType inputEmbedDim = TELayerNormF 'T5 device dataType inputEmbedDim
  TELayerNormF 'BART _ _ _ = ()
  TELayerNormF 'MBART device dataType inputEmbedDim = TELayerNormF 'BART device dataType inputEmbedDim
  TELayerNormF 'Pegasus device dataType inputEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[inputEmbedDim])
  TELayerNormF 'BERT _ _ _ = ()
  TELayerNormF 'RoBERTa device dataType inputEmbedDim = TELayerNormF 'BERT device dataType inputEmbedDim

type family
  TEDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  TEDropoutF _ dropoutP = Dropout dropoutP

type family
  TEPosEncF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEPosEncF 'T5 device dataType headDim _ posEncDim = Embedding ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TEPosEncF 'ByT5 device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'T5 device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'BART device dataType _ inputEmbedDim posEncDim = Embedding ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing
  TEPosEncF 'MBART device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BART device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'Pegasus device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BART device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'BERT device dataType _ inputEmbedDim posEncDim = Embedding ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing
  TEPosEncF 'RoBERTa device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BERT device dataType headDim inputEmbedDim posEncDim

type family
  HasInitializeTEEmbedLayerNormInputF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTEEmbedLayerNormInputF 'T5 _ _ _ = ()
  HasInitializeTEEmbedLayerNormInputF 'ByT5 device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'T5 device dataType inputEmbedDim
  HasInitializeTEEmbedLayerNormInputF 'BART device dataType inputEmbedDim = (SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTEEmbedLayerNormInputF 'MBART device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'BART device dataType inputEmbedDim
  HasInitializeTEEmbedLayerNormInputF 'Pegasus _ _ _ = ()
  HasInitializeTEEmbedLayerNormInputF 'BERT device dataType inputEmbedDim = (SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTEEmbedLayerNormInputF 'RoBERTa device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'BERT device dataType inputEmbedDim

type family
  HasInitializeTELayerNormInputF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTELayerNormInputF 'T5 device dataType inputEmbedDim = (SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTELayerNormInputF 'ByT5 device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'T5 device dataType inputEmbedDim
  HasInitializeTELayerNormInputF 'BART _ _ _ = ()
  HasInitializeTELayerNormInputF 'MBART device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'BART device dataType inputEmbedDim
  HasInitializeTELayerNormInputF 'Pegasus device dataType inputEmbedDim = (SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTELayerNormInputF 'BERT _ _ _ = ()
  HasInitializeTELayerNormInputF 'RoBERTa device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'BERT device dataType inputEmbedDim

type family
  HasInitializeTEPosEncInputF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTEPosEncInputF 'T5 device dataType headDim _ posEncDim = (SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim headDim)
  HasInitializeTEPosEncInputF 'ByT5 device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'T5 device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'BART device dataType _ inputEmbedDim posEncDim = (SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim inputEmbedDim)
  HasInitializeTEPosEncInputF 'MBART device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BART device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'Pegasus device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BART device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'BERT device dataType _ inputEmbedDim posEncDim = (SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim inputEmbedDim)
  HasInitializeTEPosEncInputF 'RoBERTa device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BERT device dataType headDim inputEmbedDim posEncDim

instance
  ( SingI style,
    stack ~ TEStackF numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    HasInitialize stack (SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, dropoutP, Double) generator generator',
    embedLayerNorm ~ TEEmbedLayerNormF style device dataType inputEmbedDim,
    HasInitialize embedLayerNorm (HasInitializeTEEmbedLayerNormInputF style device dataType inputEmbedDim) generator' generator'',
    layerNorm ~ TELayerNormF style device dataType inputEmbedDim,
    HasInitialize layerNorm (HasInitializeTELayerNormInputF style device dataType inputEmbedDim) generator'' generator''',
    dropout ~ TEDropoutF style dropoutP,
    HasInitialize dropout dropoutP generator''' generator''',
    posEnc ~ TEPosEncF style device dataType headDim inputEmbedDim posEncDim,
    HasInitialize posEnc (HasInitializeTEPosEncInputF style device dataType headDim inputEmbedDim posEncDim) generator''' generator''''
  ) =>
  HasInitialize
    (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
    generator
    generator''''
  where
  initialize (device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) =
    let stack = IxState . initialize $ (device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps)
        embedLayerNorm = IxState . initialize $ case sing @style of
          ST5 -> ()
          SByT5 -> ()
          SBART -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SMBART -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SPegasus -> ()
          SBERT -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SRoBERTa -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SGPT2 -> undefined
        layerNorm = IxState . initialize $ case sing @style of
          ST5 -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SByT5 -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SBART -> ()
          SMBART -> ()
          SPegasus -> (device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SBERT -> ()
          SRoBERTa -> ()
          SGPT2 -> undefined
        dropout = IxState . initialize $ dropoutP
        posEnc = IxState . initialize $ case sing @style of
          ST5 -> (SLayout SDense, device, dataType, posEncDim, headDim)
          SByT5 -> (SLayout SDense, device, dataType, posEncDim, headDim)
          SBART -> (SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SMBART -> (SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SPegasus -> (SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SBERT -> (SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SRoBERTa -> (SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SGPT2 -> undefined
     in runIxState $
          ( GTransformerEncoder
              <<$>> stack
              <<*>> embedLayerNorm
              <<*>> layerNorm
              <<*>> dropout
              <<*>> posEnc
          )
            >>>= ireturn . TransformerEncoder

lookupEncoder ::
  forall numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim embedDim,
    KnownDim inputEmbedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    Scalar dropoutP
    -- HasLookupStack numLayers (1 <=? numLayers) numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  Double ->
  String ->
  m (TransformerEncoder numLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
lookupEncoder headDim headEmbedDim embedDim dropoutP eps prefix =
  let stack ST5 = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "block.")
      stack SByT5 = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "block.")
      stack SBART = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SMBART = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SPegasus = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "layers.")
      stack SBERT = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "encoder.layer.")
      stack SRoBERTa = undefined headDim headEmbedDim embedDim dropoutP eps (prefix <> "encoder.layer.")
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
      embedLayerNorm SBERT =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "embeddings.LayerNorm.weight")
          <*> lookupTensor (prefix <> "embeddings.LayerNorm.bias")
          <*> pure eps
      embedLayerNorm SRoBERTa =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "embeddings.LayerNorm.weight")
          <*> lookupTensor (prefix <> "embeddings.LayerNorm.bias")
          <*> pure eps
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
      layerNorm SBERT = pure ()
      layerNorm SRoBERTa = pure ()
      layerNorm SGPT2 = undefined
      dropout _ = pure (Dropout dropoutP)
      posEnc ST5 = fmap @m Embedding $ lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
      posEnc SByT5 = fmap @m Embedding $ lookupTensor (prefix <> "block.0.layer.0.SelfAttention.relative_attention_bias.weight")
      posEnc SBART = fmap @m Embedding $ lookupTensor (prefix <> "embed_positions.weight")
      posEnc SMBART = fmap @m Embedding $ lookupTensor (prefix <> "embed_positions.weight")
      posEnc SPegasus = fmap @m Embedding $ lookupTensor (prefix <> "embed_positions.weight")
      posEnc SBERT = fmap @m Embedding $ lookupTensor (prefix <> "embeddings.position_embeddings.weight")
      posEnc SRoBERTa = fmap @m Embedding $ lookupTensor (prefix <> "embeddings.position_embeddings.weight")
      posEnc SGPT2 = undefined
   in TransformerEncoder
        <$> ( GTransformerEncoder
                <$> stack (sing @style)
                <*> embedLayerNorm (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
                <*> posEnc (sing @style)
            )

-- | 'HasForward' instance for @TransformerEncoder numLayers 'T5@.
--
-- @
--  ┌───────┐  ┌────────┐  ┌───────────────┐
--  │ input │  │ relPos │  │ attentionMask │
--  └───┬───┘  └───┬────┘  └───────┬───────┘
--      │          │               │
--      │          ▼               │
--      │      tePosEnc            │
--      │          ▼               │
--      │      transpose           │
--      │          ▼               ▼
--      │      transpose       unsqueeze
--      ▼          │               │
--  teDropout      └─────►add◄─────┘
--      ▼                  │
--   teStack◄──────────────┘
--      ▼
-- teLayerNorm
--      ▼
--  teDropout
--      │
--      ▼
--  ┌────────┐
--  │ output │
--  └────────┘
-- @
instance
  ( HasForward
      (TEDropoutF 'T5 dropoutP)
      input
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          'WithGradient
          ('Layout 'Dense <+> relPosLayout <+> attentionMaskLayout)
          (device <+> relPosDevice <+> attentionMaskDevice)
          (Seq (relPosDataType <+> 'DataType 'Int64) dataType <+> attentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          relPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  attentionMaskShape
              )
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      (TELayerNormF 'T5 device dataType inputEmbedDim)
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (TEDropoutF 'T5 dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( input,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxState $
          ireturn input
            >>>= IxState . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxState $ forward teStack (input', attentionBias')))
            >>>= IxState . forward teLayerNorm
            >>>= IxState . forward teDropout

testEncoder = do
  let device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      posEncDim = SName @"*" :&: SSize @32
      dropoutP :: Float = 0.0
      eps = 1e-6
  g <- sMkGenerator device 0
  let (encoder, g') = initialize @(TransformerEncoder 10 'T5 _ _ _ _ _ _ _ _ _) (device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = sOnes SWithoutGradient (SLayout SDense) device
      input = sOnes' dataType (SShape $ batchDim :|: seqDim :|: inputEmbedDim :|: SNil)
      relPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      attentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  let (output, _) = forward encoder (input, relPos, attentionMask) g'
  pure output

-- | 'HasForward' instance for @TransformerEncoder numLayers 'ByT5@.
--
-- @
--  ┌───────┐  ┌────────┐  ┌───────────────┐
--  │ input │  │ relPos │  │ attentionMask │
--  └───┬───┘  └───┬────┘  └───────┬───────┘
--      │          │               │
--      │          ▼               │
--      │      tePosEnc            │
--      │          ▼               │
--      │      transpose           │
--      │          ▼               ▼
--      │      transpose       unsqueeze
--      ▼          │               │
--  teDropout      └─────►add◄─────┘
--      ▼                  │
--   teStack◄──────────────┘
--      ▼
-- teLayerNorm
--      ▼
--  teDropout
--      │
--      ▼
--  ┌────────┐
--  │ output │
--  └────────┘
-- @
instance
  ( HasForward
      (TEDropoutF 'ByT5 dropoutP)
      input
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'ByT5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          'WithGradient
          ('Layout 'Dense <+> relPosLayout <+> attentionMaskLayout)
          (device <+> relPosDevice <+> attentionMaskDevice)
          (Seq (relPosDataType <+> 'DataType 'Int64) dataType <+> attentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( EmbeddingF
                          ('Shape '[posEncDim, headDim])
                          relPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ('SelectDim ('ByIndex 1))
                  attentionMaskShape
              )
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      (TELayerNormF 'ByT5 device dataType inputEmbedDim)
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (TEDropoutF 'ByT5 dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'ByT5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( input,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxState $
          ireturn input
            >>>= IxState . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxState $ forward teStack (input', attentionBias')))
            >>>= IxState . forward teLayerNorm
            >>>= IxState . forward teDropout

-- | 'HasForward' instance for @TransformerEncoder numLayers 'BART@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEEmbedLayerNormF 'BART device dataType inputEmbedDim)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      layerNormOutput
      generator,
    HasForward
      (TEDropoutF 'BART dropoutP)
      layerNormOutput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'BART device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'BART device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teEmbedLayerNorm
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))

-- | 'HasForward' instance for @TransformerEncoder numLayers 'MBART@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          ▼
--     teLayerNorm
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @

-- | 'HasForward' instance for @TransformerEncoder numLayers 'BERT@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEEmbedLayerNormF 'BERT device dataType inputEmbedDim)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      layerNormOutput
      generator,
    HasForward
      (TEDropoutF 'BERT dropoutP)
      layerNormOutput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'BERT device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'BERT device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teEmbedLayerNorm
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))

-- | 'HasForward' instance for @TransformerEncoder numLayers 'RoBERTa@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
--   teEmbedLayerNorm          │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEEmbedLayerNormF 'RoBERTa device dataType inputEmbedDim)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      layerNormOutput
      generator,
    HasForward
      (TEDropoutF 'RoBERTa dropoutP)
      layerNormOutput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'RoBERTa device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerEncoder numLayers 'RoBERTa device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teEmbedLayerNorm
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))

-- | 'HasForward' instance for @TransformerEncoder numLayers 'Pegasus@.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │     tePosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  ▼
--      teDropout          unsqueeze
--          ▼                  │
--       teStack◄──────────────┘
--          ▼
--     teLayerNorm
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      (TEDropoutF 'Pegasus dropoutP)
      ( Tensor
          'WithGradient
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'Pegasus device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TELayerNormF 'Pegasus device dataType inputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput,
    Show output
  ) =>
  HasForward
    (TransformerEncoder numLayers 'Pegasus device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posRequiresGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxState $
          ireturn pos
            >>>= IxState . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxState . forward teDropout
            >>>= (\input' -> IxState $ forward teStack (input', attentionBias))
            >>>= IxState . forward teLayerNorm
