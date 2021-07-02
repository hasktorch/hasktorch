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

module Torch.GraduallyTyped.NN.Transformer.Encoder where

import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (KnownNat, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

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
    (gradient :: Gradient RequiresGradient)
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
    forall numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP.
    GTransformerEncoderF numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP ->
    TransformerEncoder numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type GTransformerEncoderF
  (numLayers :: Nat)
  (style :: TransformerStyle)
  (gradient :: Gradient RequiresGradient)
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
    (TEStackF numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
    (TEEmbedLayerNormF style gradient device dataType inputEmbedDim)
    (TELayerNormF style gradient device dataType inputEmbedDim)
    (TEDropoutF style dropoutP)
    (TEPosEncF style gradient device dataType headDim inputEmbedDim posEncDim)

type family
  TEStackF
    (numLayers :: Nat)
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
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
  TEStackF numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP =
    TransformerStack numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP

type family
  TEEmbedLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEEmbedLayerNormF 'T5 _ _ _ _ = ()
  TEEmbedLayerNormF 'ByT5 gradient device dataType inputEmbedDim = TEEmbedLayerNormF 'T5 gradient device dataType inputEmbedDim
  TEEmbedLayerNormF 'BART gradient device dataType inputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim])
  TEEmbedLayerNormF 'MBART gradient device dataType inputEmbedDim = TEEmbedLayerNormF 'BART gradient device dataType inputEmbedDim
  TEEmbedLayerNormF 'Pegasus _ _ _ _ = ()
  TEEmbedLayerNormF 'BERT gradient device dataType inputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim])
  TEEmbedLayerNormF 'RoBERTa gradient device dataType inputEmbedDim = TEEmbedLayerNormF 'BERT gradient device dataType inputEmbedDim

type family
  TELayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TELayerNormF 'T5 gradient device dataType inputEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[inputEmbedDim])
  TELayerNormF 'ByT5 gradient device dataType inputEmbedDim = TELayerNormF 'T5 gradient device dataType inputEmbedDim
  TELayerNormF 'BART _ _ _ _ = ()
  TELayerNormF 'MBART gradient device dataType inputEmbedDim = TELayerNormF 'BART gradient device dataType inputEmbedDim
  TELayerNormF 'Pegasus gradient device dataType inputEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim])
  TELayerNormF 'BERT _ _ _ _ = ()
  TELayerNormF 'RoBERTa gradient device dataType inputEmbedDim = TELayerNormF 'BERT gradient device dataType inputEmbedDim

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
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEPosEncF 'T5 gradient device dataType headDim _ posEncDim = Embedding gradient ('Layout 'Dense) device dataType posEncDim headDim 'Nothing
  TEPosEncF 'ByT5 gradient device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'T5 gradient device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'BART gradient device dataType _ inputEmbedDim posEncDim = Embedding gradient ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing
  TEPosEncF 'MBART gradient device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BART gradient device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'Pegasus gradient device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BART gradient device dataType headDim inputEmbedDim posEncDim
  TEPosEncF 'BERT gradient device dataType _ inputEmbedDim posEncDim = Embedding gradient ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing
  TEPosEncF 'RoBERTa gradient device dataType headDim inputEmbedDim posEncDim = TEPosEncF 'BERT gradient device dataType headDim inputEmbedDim posEncDim

type family
  HasInitializeTEEmbedLayerNormInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTEEmbedLayerNormInputF 'T5 _ _ _ _ = ()
  HasInitializeTEEmbedLayerNormInputF 'ByT5 gradient device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'T5 gradient device dataType inputEmbedDim
  HasInitializeTEEmbedLayerNormInputF 'BART gradient device dataType inputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTEEmbedLayerNormInputF 'MBART gradient device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'BART gradient device dataType inputEmbedDim
  HasInitializeTEEmbedLayerNormInputF 'Pegasus _ _ _ _ = ()
  HasInitializeTEEmbedLayerNormInputF 'BERT gradient device dataType inputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTEEmbedLayerNormInputF 'RoBERTa gradient device dataType inputEmbedDim = HasInitializeTEEmbedLayerNormInputF 'BERT gradient device dataType inputEmbedDim

type family
  HasInitializeTELayerNormInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTELayerNormInputF 'T5 gradient device dataType inputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTELayerNormInputF 'ByT5 gradient device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'T5 gradient device dataType inputEmbedDim
  HasInitializeTELayerNormInputF 'BART _ _ _ _ = ()
  HasInitializeTELayerNormInputF 'MBART gradient device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'BART gradient device dataType inputEmbedDim
  HasInitializeTELayerNormInputF 'Pegasus gradient device dataType inputEmbedDim = (SGradient gradient, SDevice device, SDataType dataType, SShape ('Shape '[inputEmbedDim]), Double)
  HasInitializeTELayerNormInputF 'BERT _ _ _ _ = ()
  HasInitializeTELayerNormInputF 'RoBERTa gradient device dataType inputEmbedDim = HasInitializeTELayerNormInputF 'BERT gradient device dataType inputEmbedDim

type family
  HasInitializeTEPosEncInputF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeTEPosEncInputF 'T5 gradient device dataType headDim _ posEncDim = (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim headDim)
  HasInitializeTEPosEncInputF 'ByT5 gradient device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'T5 gradient device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'BART gradient device dataType _ inputEmbedDim posEncDim = (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim inputEmbedDim)
  HasInitializeTEPosEncInputF 'MBART gradient device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BART gradient device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'Pegasus gradient device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BART gradient device dataType headDim inputEmbedDim posEncDim
  HasInitializeTEPosEncInputF 'BERT gradient device dataType _ inputEmbedDim posEncDim = (SGradient gradient, SLayout ('Layout 'Dense), SDevice device, SDataType dataType, SDim posEncDim, SDim inputEmbedDim)
  HasInitializeTEPosEncInputF 'RoBERTa gradient device dataType headDim inputEmbedDim posEncDim = HasInitializeTEPosEncInputF 'BERT gradient device dataType headDim inputEmbedDim posEncDim

instance
  ( SingI style,
    stack ~ TEStackF numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    HasInitialize stack (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, dropoutP, Double) generator generator',
    embedLayerNorm ~ TEEmbedLayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize embedLayerNorm (HasInitializeTEEmbedLayerNormInputF style gradient device dataType inputEmbedDim) generator' generator'',
    layerNorm ~ TELayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize layerNorm (HasInitializeTELayerNormInputF style gradient device dataType inputEmbedDim) generator'' generator''',
    dropout ~ TEDropoutF style dropoutP,
    HasInitialize dropout dropoutP generator''' generator''',
    posEnc ~ TEPosEncF style gradient device dataType headDim inputEmbedDim posEncDim,
    HasInitialize posEnc (HasInitializeTEPosEncInputF style gradient device dataType headDim inputEmbedDim posEncDim) generator''' generator''''
  ) =>
  HasInitialize
    (TransformerEncoder numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
    generator
    generator''''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) =
    let stack = IxState . initialize $ (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps)
        embedLayerNorm = IxState . initialize $ case sing @style of
          ST5 -> ()
          SByT5 -> ()
          SBART -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SMBART -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SPegasus -> ()
          SBERT -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SRoBERTa -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SGPT2 -> undefined
        layerNorm = IxState . initialize $ case sing @style of
          ST5 -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SByT5 -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SBART -> ()
          SMBART -> ()
          SPegasus -> (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps)
          SBERT -> ()
          SRoBERTa -> ()
          SGPT2 -> undefined
        dropout = IxState . initialize $ dropoutP
        posEnc = IxState . initialize $ case sing @style of
          ST5 -> (gradient, SLayout SDense, device, dataType, posEncDim, headDim)
          SByT5 -> (gradient, SLayout SDense, device, dataType, posEncDim, headDim)
          SBART -> (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SMBART -> (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SPegasus -> (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SBERT -> (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
          SRoBERTa -> (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim)
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

instance
  (SingI style, KnownNat numLayers) =>
  HasStateDict
    (TransformerEncoder numLayers style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim posEncDim, dropoutP, Double)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) k =
    let stack ST5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "block.")
        stack SByT5 = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "block.")
        stack SBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SMBART = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SPegasus = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "layers.")
        stack SBERT = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "encoder.layer.")
        stack SRoBERTa = fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, dropoutP, eps) (k <> "encoder.layer.")
        stack SGPT2 = undefined
        embedLayerNorm ST5 = fromStateDict () k
        embedLayerNorm SByT5 = fromStateDict () k
        embedLayerNorm SBART = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = fromStateDict () k
        embedLayerNorm SBERT = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "embeddings.LayerNorm.")
        embedLayerNorm SRoBERTa = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "embeddings.LayerNorm.")
        embedLayerNorm SGPT2 = undefined
        layerNorm ST5 = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SByT5 = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "final_layer_norm.")
        layerNorm SBART = fromStateDict () k
        layerNorm SMBART = fromStateDict () k
        layerNorm SPegasus = fromStateDict (gradient, device, dataType, SShape $ inputEmbedDim :|: SNil, eps) (k <> "layer_norm.")
        layerNorm SBERT = fromStateDict () k
        layerNorm SRoBERTa = fromStateDict () k
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
        posEnc ST5 = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, headDim) (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, headDim) (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim) (k <> "embed_positions.")
        posEnc SMBART = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim) (k <> "embed_positions.")
        posEnc SPegasus = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim) (k <> "embed_positions.")
        posEnc SBERT = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim) (k <> "pos_embed.")
        posEnc SRoBERTa = fromStateDict (gradient, SLayout SDense, device, dataType, posEncDim, inputEmbedDim) (k <> "pos_embed.")
        posEnc SGPT2 = undefined
     in TransformerEncoder
          <$> ( GTransformerEncoder
                  <$> stack (sing @style)
                  <*> embedLayerNorm (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
                  <*> posEnc (sing @style)
              )
  toStateDict k (TransformerEncoder GTransformerEncoder {..}) =
    let stack ST5 = toStateDict (k <> "block.")
        stack SByT5 = toStateDict (k <> "block.")
        stack SBART = toStateDict (k <> "layers.")
        stack SMBART = toStateDict (k <> "layers.")
        stack SPegasus = toStateDict (k <> "layers.")
        stack SBERT = toStateDict (k <> "encoder.layer.")
        stack SRoBERTa = toStateDict (k <> "encoder.layer.")
        stack SGPT2 = undefined
        embedLayerNorm ST5 = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SByT5 = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SBART = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = toStateDict (k <> "layernorm_embedding.")
        embedLayerNorm SBERT = toStateDict (k <> "embeddings.LayerNorm.")
        embedLayerNorm SRoBERTa = toStateDict (k <> "embeddings.LayerNorm.")
        embedLayerNorm SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "final_layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "final_layer_norm.")
        layerNorm SBART = toStateDict (k <> "final_layer_norm.")
        layerNorm SMBART = toStateDict (k <> "final_layer_norm.")
        layerNorm SPegasus = toStateDict (k <> "layernorm_embedding.")
        layerNorm SBERT = toStateDict (k <> "layernorm_embedding.")
        layerNorm SRoBERTa = toStateDict (k <> "layernorm_embedding.")
        layerNorm SGPT2 = undefined
        dropout _ = toStateDict k
        posEnc ST5 = toStateDict (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = toStateDict (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = toStateDict (k <> "embed_positions.")
        posEnc SMBART = toStateDict (k <> "embed_positions.")
        posEnc SPegasus = toStateDict (k <> "embed_positions.")
        posEnc SBERT = toStateDict (k <> "pos_embed.")
        posEnc SRoBERTa = toStateDict (k <> "pos_embed.")
        posEnc SGPT2 = undefined
     in do
          () <- stack (sing @style) teStack
          () <- embedLayerNorm (sing @style) teEmbedLayerNorm
          () <- layerNorm (sing @style) teLayerNorm
          () <- dropout (sing @style) teDropout
          () <- posEnc (sing @style) tePosEnc
          pure ()

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
      (TEStackF numLayers 'T5 gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          (gradient <|> relPosGradient <|> attentionMaskGradient)
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
      (TELayerNormF 'T5 gradient device dataType inputEmbedDim)
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
    (TransformerEncoder numLayers 'T5 gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( input,
      Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
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
  let (encoder, g') = initialize @(TransformerEncoder 10 'T5 _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, posEncDim, dropoutP, eps) g
      batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
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
      (TEStackF numLayers 'ByT5 gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          (gradient <|> relPosGradient <|> attentionMaskGradient)
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
      (TELayerNormF 'ByT5 gradient device dataType inputEmbedDim)
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
    (TransformerEncoder numLayers 'ByT5 gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( input,
      Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
      (TEEmbedLayerNormF 'BART gradient device dataType inputEmbedDim)
      ( Tensor
          (inputGradient <|> gradient <|> posGradient)
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
      (TEStackF numLayers 'BART gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
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
    (TransformerEncoder numLayers 'BART gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
      (TEEmbedLayerNormF 'BERT gradient device dataType inputEmbedDim)
      ( Tensor
          (inputGradient <|> gradient <|> posGradient)
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
      (TEStackF numLayers 'BERT gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
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
    (TransformerEncoder numLayers 'BERT gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
      (TEEmbedLayerNormF 'RoBERTa gradient device dataType inputEmbedDim)
      ( Tensor
          (inputGradient <|> gradient <|> posGradient)
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
      (TEStackF numLayers 'RoBERTa gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
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
    (TransformerEncoder numLayers 'RoBERTa gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
          (inputGradient <|> gradient <|> posGradient)
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TEStackF numLayers 'Pegasus gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutput
      stackOutput
      generatorOutput,
    HasForward
      (TELayerNormF 'Pegasus gradient device dataType inputEmbedDim)
      stackOutput
      generatorOutput
      output
      generatorOutput,
    Show output
  ) =>
  HasForward
    (TransformerEncoder numLayers 'Pegasus gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
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
