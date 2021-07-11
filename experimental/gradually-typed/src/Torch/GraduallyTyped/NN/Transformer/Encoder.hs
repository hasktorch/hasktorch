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

module Torch.GraduallyTyped.NN.Transformer.Encoder where

import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.Prelude.List (SList (SNil))
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack, TransformerStackSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (SWithBias, SWithoutBias))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
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
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
  where
  TransformerEncoder ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim.
    GTransformerEncoder
      (TEStackF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
      (TEEmbedLayerNormF style gradient device dataType inputEmbedDim)
      (TELayerNormF style gradient device dataType inputEmbedDim)
      (TEDropoutF style)
      (TEPosEncF style gradient device dataType headDim inputEmbedDim posEncDim) ->
    TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim

data
  TransformerEncoderSpec
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
  where
  TransformerEncoderSpec ::
    forall style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim.
    STransformerStyle style ->
    SNat numLayers ->
    SGradient gradient ->
    SDevice device ->
    SDataType dataType ->
    SDim headDim ->
    SDim headEmbedDim ->
    SDim embedDim ->
    SDim inputEmbedDim ->
    SDim ffnDim ->
    SDim posEncDim ->
    Double ->
    Double ->
    TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim

type instance ModelSpec (TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim) = TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim

type family
  TEStackF
    (style :: TransformerStyle)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEStackF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim =
    TransformerStack style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim

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
    (style :: TransformerStyle) ::
    Type
  where
  TEDropoutF _ = Dropout

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

instance
  ( SingI style,
    stack ~ TEStackF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim,
    HasInitialize stack device stack device,
    embedLayerNorm ~ TEEmbedLayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize embedLayerNorm device embedLayerNorm device,
    layerNorm ~ TELayerNormF style gradient device dataType inputEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    dropout ~ TEDropoutF style,
    HasInitialize dropout device dropout device,
    posEnc ~ TEPosEncF style gradient device dataType headDim inputEmbedDim posEncDim,
    HasInitialize posEnc device posEnc device
  ) =>
  HasInitialize
    (TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    generatorDevice
    (TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    device
  where
  initialize (TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        stack = IxStateT . initialize @stack $ TransformerStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP eps
        embedLayerNormSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        embedLayerNorm = IxStateT . initialize @embedLayerNorm $ case sing @style of
          ST5 -> ()
          SByT5 -> ()
          SBART -> embedLayerNormSpec
          SMBART -> embedLayerNormSpec
          SPegasus -> ()
          SBERT -> embedLayerNormSpec
          SRoBERTa -> embedLayerNormSpec
          SGPT2 -> undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $ case sing @style of
          ST5 -> layerNormWithoutBiasSpec
          SByT5 -> layerNormWithoutBiasSpec
          SBART -> ()
          SMBART -> ()
          SPegasus -> layerNormWithBiasSpec
          SBERT -> ()
          SRoBERTa -> ()
          SGPT2 -> undefined
        dropout = IxStateT . initialize @dropout $ Dropout dropoutP
        relPosEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
        posEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim inputEmbedDim SNothing
        posEnc = IxStateT . initialize @posEnc $ case sing @style of
          ST5 -> relPosEncSpec
          SByT5 -> relPosEncSpec
          SBART -> posEncSpec
          SMBART -> posEncSpec
          SPegasus -> posEncSpec
          SBERT -> posEncSpec
          SRoBERTa -> posEncSpec
          SGPT2 -> undefined
        gte =
          GTransformerEncoder
            <<$>> stack
            <<*>> embedLayerNorm
            <<*>> layerNorm
            <<*>> dropout
            <<*>> posEnc
     in runIxStateT (gte >>>= ireturn . TransformerEncoder) generator'

instance
  SingI style =>
  HasStateDict
    (TransformerEncoder style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
  where
  fromStateDict (TransformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps) k =
    let stackSpec = TransformerStackSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP eps
        stack ST5 = fromStateDict stackSpec (k <> "block.")
        stack SByT5 = fromStateDict stackSpec (k <> "block.")
        stack SBART = fromStateDict stackSpec (k <> "layers.")
        stack SMBART = fromStateDict stackSpec (k <> "layers.")
        stack SPegasus = fromStateDict stackSpec (k <> "layers.")
        stack SBERT = fromStateDict stackSpec (k <> "encoder.layer.")
        stack SRoBERTa = fromStateDict stackSpec (k <> "encoder.layer.")
        stack SGPT2 = undefined
        embedLayerNormSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        embedLayerNorm ST5 = fromStateDict () k
        embedLayerNorm SByT5 = fromStateDict () k
        embedLayerNorm SBART = fromStateDict embedLayerNormSpec (k <> "layernorm_embedding.")
        embedLayerNorm SMBART = fromStateDict embedLayerNormSpec (k <> "layernorm_embedding.")
        embedLayerNorm SPegasus = fromStateDict () k
        embedLayerNorm SBERT = fromStateDict embedLayerNormSpec (k <> "embeddings.LayerNorm.")
        embedLayerNorm SRoBERTa = fromStateDict embedLayerNormSpec (k <> "embeddings.LayerNorm.")
        embedLayerNorm SGPT2 = undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
        layerNorm ST5 = fromStateDict layerNormWithoutBiasSpec (k <> "final_layer_norm.")
        layerNorm SByT5 = fromStateDict layerNormWithoutBiasSpec (k <> "final_layer_norm.")
        layerNorm SBART = fromStateDict () k
        layerNorm SMBART = fromStateDict () k
        layerNorm SPegasus = fromStateDict layerNormWithBiasSpec (k <> "layer_norm.")
        layerNorm SBERT = fromStateDict () k
        layerNorm SRoBERTa = fromStateDict () k
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict (Dropout dropoutP) k
        relPosEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
        posEncSpec = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim inputEmbedDim SNothing
        posEnc ST5 = fromStateDict relPosEncSpec (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SByT5 = fromStateDict relPosEncSpec (k <> "block.0.layer.0.SelfAttention.relative_attention_bias.")
        posEnc SBART = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SMBART = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SPegasus = fromStateDict posEncSpec (k <> "embed_positions.")
        posEnc SBERT = fromStateDict posEncSpec (k <> "embeddings.position_embeddings.")
        posEnc SRoBERTa = fromStateDict posEncSpec (k <> "embeddings.position_embeddings.")
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
        posEnc SBERT = toStateDict (k <> "embeddings.position_embeddings.")
        posEnc SRoBERTa = toStateDict (k <> "embeddings.position_embeddings.")
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
      (TEDropoutF 'T5)
      input
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      stackOutput
      stackGeneratorOutputDevice,
    HasForward
      (TELayerNormF 'T5 gradient device dataType inputEmbedDim)
      stackOutput
      stackGeneratorOutputDevice
      layerNormOutput
      layerNormGeneratorOutputDevice,
    HasForward
      (TEDropoutF 'T5)
      layerNormOutput
      layerNormGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerEncoder 'T5 numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( input,
      Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxStateT . forward tePosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxStateT $ forward teStack (input', attentionBias')))
            >>>= IxStateT . forward teLayerNorm
            >>>= IxStateT . forward teDropout

testEncoder :: IO _
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
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (encoder, g') <- initialize (TransformerEncoderSpec ST5 (SNat @10) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps) g
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' dataType (SShape $ batchDim :|: seqDim :|: inputEmbedDim :|: SNil)
      relPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      attentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoder (input, relPos, attentionMask) g'
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
      (TEDropoutF 'ByT5)
      input
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
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
      dropoutGeneratorOutputDevice
      stackOutput
      stackGeneratorOutputDevice,
    HasForward
      (TELayerNormF 'ByT5 gradient device dataType inputEmbedDim)
      stackOutput
      stackGeneratorOutputDevice
      layerNormOutput
      layerNormGeneratorOutputDevice,
    HasForward
      (TEDropoutF 'ByT5)
      layerNormOutput
      layerNormGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerEncoder 'ByT5 numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( input,
      Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxStateT . forward tePosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxStateT $ forward teStack (input', attentionBias')))
            >>>= IxStateT . forward teLayerNorm
            >>>= IxStateT . forward teDropout

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
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (TEDropoutF 'BART)
      layerNormOutput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'BART numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerEncoder 'BART numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxStateT $
          ireturn pos
            >>>= IxStateT . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxStateT . forward teEmbedLayerNorm
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> IxStateT $ forward teStack (input', attentionBias))

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
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (TEDropoutF 'BERT)
      layerNormOutput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'BERT numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerEncoder 'BERT numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxStateT $
          ireturn pos
            >>>= IxStateT . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxStateT . forward teEmbedLayerNorm
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> IxStateT $ forward teStack (input', attentionBias))

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
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (TEDropoutF 'RoBERTa)
      layerNormOutput
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'RoBERTa numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutputDevice
      output
      generatorOutputDevice
  ) =>
  HasForward
    (TransformerEncoder 'RoBERTa numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxStateT $
          ireturn pos
            >>>= IxStateT . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxStateT . forward teEmbedLayerNorm
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> IxStateT $ forward teStack (input', attentionBias))

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
      (TEDropoutF 'Pegasus)
      ( Tensor
          (inputGradient <|> gradient <|> posGradient)
          (inputLayout <+> 'Layout 'Dense <+> posLayout)
          (inputDevice <+> device <+> posDevice)
          (inputDataType <+> Seq (posDataType <+> 'DataType 'Int64) dataType)
          (BroadcastShapesF inputShape (EmbeddingF ('Shape '[posEncDim, inputEmbedDim]) posShape))
      )
      generatorDevice
      dropoutOutput
      dropoutGeneratorOutputDevice,
    HasForward
      (TEStackF 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
      ( dropoutOutput,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      dropoutGeneratorOutputDevice
      stackOutput
      generatorOutputDevice,
    HasForward
      (TELayerNormF 'Pegasus gradient device dataType inputEmbedDim)
      stackOutput
      generatorOutputDevice
      output
      generatorOutputDevice,
    Show output
  ) =>
  HasForward
    (TransformerEncoder 'Pegasus numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (TransformerEncoder GTransformerEncoder {..}) (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxStateT $
          ireturn pos
            >>>= IxStateT . forward tePosEnc
            >>>= ireturn . (input `add`)
            >>>= IxStateT . forward teDropout
            >>>= (\input' -> IxStateT $ forward teStack (input', attentionBias))
            >>>= IxStateT . forward teLayerNorm
