{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
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
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.GTransformer where

import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed (IxPointed (ireturn), (<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map as Map
import Data.Singletons.Prelude.List (SList (SNil))
import Data.Singletons.Prelude.Maybe (SMaybe (SNothing))
import Data.Singletons.TypeLits (SNat (..))
import qualified Data.Vector.Sized as VS
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..), EmbeddingSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GBlock (DecoderBlockCrossAttentionF, DecoderBlockFeedForwardNetworkF, DecoderBlockSelfAttentionF, EncoderBlockCrossAttentionF, EncoderBlockFeedForwardNetworkF, EncoderBlockSelfAttentionF, GTransformerBlock)
import Torch.GraduallyTyped.NN.Transformer.GStack (GTransformerStack, decoderStackSpec, encoderStackSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (SWithBias, SWithoutBias))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic transformer.
-- Can specialize to either encoder or decoder.
--
-- - @posEnc@: an absolute positional encoding layer as used by, e.g., BERT.
-- - @relPosEnc@: a relative positional encoding layer as used by, e.g., T5.
-- - @initialLayerNorm@: a layer normalization layer for the embeddings.
-- - @initialDropout@: a dropout layer for the embeddings.
-- - @stack@: a stack of transformer blocks.
-- - @finalLayerNorm@: the final layer normalization layer.
-- - @finalDropout@: the final dropout layer.
data
  GTransformer
    (posEnc :: Type)
    (relPosEnc :: Type)
    (initialLayerNorm :: Type)
    (initialDropout :: Type)
    (stack :: Type)
    (finalLayerNorm :: Type)
    (finalDropout :: Type)
  where
  GTransformer ::
    forall posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout.
    { -- | absolute positional encoding
      tPosEnc :: posEnc,
      -- | relative positional encoding
      tRelPosEnc :: relPosEnc,
      -- | initial layer norm
      tInitialLayerNorm :: initialLayerNorm,
      -- | initial dropout
      tInitialDropout :: initialDropout,
      -- | transformer block stack
      tStack :: stack,
      -- | final layer norm
      tFinalLayerNorm :: finalLayerNorm,
      -- | final dropout
      tFinalDropout :: finalDropout
    } ->
    GTransformer posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout

type instance
  ModelSpec (GTransformer posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout) =
    GTransformer (ModelSpec posEnc) (ModelSpec relPosEnc) (ModelSpec initialLayerNorm) (ModelSpec initialDropout) (ModelSpec stack) (ModelSpec finalLayerNorm) (ModelSpec finalDropout)

-- | Specifies the absolute positional encoding layer of a transformer encoder.
type family
  TEPosEncF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEPosEncF 'T5 _ _ _ _ _ = ()
  TEPosEncF 'ByT5 gradient device dataType inputEmbedDim posEncDim = TEPosEncF 'T5 gradient device dataType inputEmbedDim posEncDim
  TEPosEncF 'BART gradient device dataType inputEmbedDim posEncDim = NamedModel (Embedding gradient ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing)
  TEPosEncF 'MBART gradient device dataType inputEmbedDim posEncDim = TEPosEncF 'BART gradient device dataType inputEmbedDim posEncDim
  TEPosEncF 'Pegasus gradient device dataType inputEmbedDim posEncDim = TEPosEncF 'BART gradient device dataType inputEmbedDim posEncDim
  TEPosEncF 'BERT gradient device dataType inputEmbedDim posEncDim = NamedModel (Embedding gradient ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing)
  TEPosEncF 'RoBERTa gradient device dataType inputEmbedDim posEncDim = TEPosEncF 'BERT gradient device dataType inputEmbedDim posEncDim

-- | Specifies the relative positional encoding layer of a transformer encoder.
type family
  TERelPosEncF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TERelPosEncF 'T5 gradient device dataType headDim posEncDim = NamedModel (Embedding gradient ('Layout 'Dense) device dataType posEncDim headDim 'Nothing)
  TERelPosEncF 'ByT5 gradient device dataType headDim posEncDim = TERelPosEncF 'T5 gradient device dataType headDim posEncDim
  TERelPosEncF 'BART _ _ _ _ _ = ()
  TERelPosEncF 'MBART gradient device dataType headDim posEncDim = TERelPosEncF 'BART gradient device dataType headDim posEncDim
  TERelPosEncF 'Pegasus gradient device dataType headDim posEncDim = TERelPosEncF 'BART gradient device dataType headDim posEncDim
  TERelPosEncF 'BERT _ _ _ _ _ = ()
  TERelPosEncF 'RoBERTa gradient device dataType headDim posEncDim = TERelPosEncF 'BERT gradient device dataType headDim posEncDim

-- | Specifies the initial layer normalization layer of a transformer encoder.
type family
  TEInitialLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEInitialLayerNormF 'T5 _ _ _ _ = ()
  TEInitialLayerNormF 'ByT5 gradient device dataType inputEmbedDim = TEInitialLayerNormF 'T5 gradient device dataType inputEmbedDim
  TEInitialLayerNormF 'BART gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))
  TEInitialLayerNormF 'MBART gradient device dataType inputEmbedDim = TEInitialLayerNormF 'BART gradient device dataType inputEmbedDim
  TEInitialLayerNormF 'Pegasus _ _ _ _ = ()
  TEInitialLayerNormF 'BERT gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))
  TEInitialLayerNormF 'RoBERTa gradient device dataType inputEmbedDim = TEInitialLayerNormF 'BERT gradient device dataType inputEmbedDim

-- | Specifies the initial dropout layer of a transformer encoder.
type family
  TEInitialDropoutF
    (style :: TransformerStyle) ::
    Type
  where
  TEInitialDropoutF 'T5 = Dropout
  TEInitialDropoutF 'ByT5 = Dropout
  TEInitialDropoutF 'BART = Dropout
  TEInitialDropoutF 'MBART = Dropout
  TEInitialDropoutF 'Pegasus = Dropout
  TEInitialDropoutF 'BERT = Dropout
  TEInitialDropoutF 'RoBERTa = Dropout

-- | Specifies the transformer block stack of a transformer encoder.
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
    NamedModel
      ( GTransformerStack
          ( VS.Vector
              numLayers
              ( GTransformerBlock
                  (EncoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim inputEmbedDim)
                  EncoderBlockCrossAttentionF
                  (EncoderBlockFeedForwardNetworkF style gradient device dataType inputEmbedDim ffnDim)
              )
          )
      )

-- | Specifies the final layer normalization layer of a transformer encoder.
type family
  TEFinalLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TEFinalLayerNormF 'T5 gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[inputEmbedDim]))
  TEFinalLayerNormF 'ByT5 gradient device dataType inputEmbedDim = TEFinalLayerNormF 'T5 gradient device dataType inputEmbedDim
  TEFinalLayerNormF 'BART _ _ _ _ = ()
  TEFinalLayerNormF 'MBART gradient device dataType inputEmbedDim = TEFinalLayerNormF 'BART gradient device dataType inputEmbedDim
  TEFinalLayerNormF 'Pegasus gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))
  TEFinalLayerNormF 'BERT _ _ _ _ = ()
  TEFinalLayerNormF 'RoBERTa gradient device dataType inputEmbedDim = TEFinalLayerNormF 'BERT gradient device dataType inputEmbedDim

-- | Specifies the final dropout layer of a transformer encoder.
type family
  TEFinalDropoutF
    (style :: TransformerStyle) ::
    Type
  where
  TEFinalDropoutF 'T5 = Dropout
  TEFinalDropoutF 'ByT5 = Dropout
  TEFinalDropoutF 'BART = ()
  TEFinalDropoutF 'MBART = ()
  TEFinalDropoutF 'Pegasus = ()
  TEFinalDropoutF 'BERT = ()
  TEFinalDropoutF 'RoBERTa = ()

-- | Specifies the parameters of a transformer in an encoder configuration.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @inputEmbedDim@: the dimension of the transformer query embeddings.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @posEncDim@: the dimension of the positional encoding.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
transformerEncoderSpec ::
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
  ModelSpec
    ( GTransformer
        (TEPosEncF style gradient device dataType inputEmbedDim posEncDim)
        (TERelPosEncF style gradient device dataType headDim posEncDim)
        (TEInitialLayerNormF style gradient device dataType inputEmbedDim)
        (TEInitialDropoutF style)
        (TEStackF style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim)
        (TEFinalLayerNormF style gradient device dataType inputEmbedDim)
        (TEFinalDropoutF style)
    )
transformerEncoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps =
  let posEncSpec ST5 = ()
      posEncSpec SByT5 = ()
      posEncSpec SBART = NamedModel "embed_positions." posEncSpec'
      posEncSpec SMBART = NamedModel "embed_positions." posEncSpec'
      posEncSpec SPegasus = NamedModel "embed_positions." posEncSpec'
      posEncSpec SBERT = NamedModel "embeddings.position_embeddings." posEncSpec'
      posEncSpec SRoBERTa = NamedModel "embeddings.position_embeddings." posEncSpec'
      posEncSpec SGPT2 = undefined
      relPosEncSpec ST5 = NamedModel "block.0.layer.0.SelfAttention.relative_attention_bias." relPosEncSpec'
      relPosEncSpec SByT5 = NamedModel "block.0.layer.0.SelfAttention.relative_attention_bias." relPosEncSpec'
      relPosEncSpec SBART = ()
      relPosEncSpec SMBART = ()
      relPosEncSpec SPegasus = ()
      relPosEncSpec SBERT = ()
      relPosEncSpec SRoBERTa = ()
      relPosEncSpec SGPT2 = undefined
      initialLayerNormSpec ST5 = ()
      initialLayerNormSpec SByT5 = ()
      initialLayerNormSpec SBART = NamedModel "layernorm_embedding." $ layerNormSpec' SWithBias
      initialLayerNormSpec SMBART = NamedModel "layernorm_embedding." $ layerNormSpec' SWithBias
      initialLayerNormSpec SPegasus = ()
      initialLayerNormSpec SBERT = NamedModel "embeddings.LayerNorm." $ layerNormSpec' SWithBias
      initialLayerNormSpec SRoBERTa = NamedModel "embeddings.LayerNorm." $ layerNormSpec' SWithBias
      initialLayerNormSpec SGPT2 = undefined
      initialDropoutSpec ST5 = Dropout dropoutP
      initialDropoutSpec SByT5 = Dropout dropoutP
      initialDropoutSpec SBART = Dropout dropoutP
      initialDropoutSpec SMBART = Dropout dropoutP
      initialDropoutSpec SPegasus = Dropout dropoutP
      initialDropoutSpec SBERT = Dropout dropoutP
      initialDropoutSpec SRoBERTa = Dropout dropoutP
      initialDropoutSpec SGPT2 = undefined
      stackSpec ST5 = NamedModel "block." $ stackSpec' ST5
      stackSpec SByT5 = NamedModel "block." $ stackSpec' SByT5
      stackSpec SBART = NamedModel "layers." $ stackSpec' SBART
      stackSpec SMBART = NamedModel "layers." $ stackSpec' SMBART
      stackSpec SPegasus = NamedModel "layers." $ stackSpec' SPegasus
      stackSpec SBERT = NamedModel "encoder.layer." $ stackSpec' SBERT
      stackSpec SRoBERTa = NamedModel "encoder.layer." $ stackSpec' SRoBERTa
      stackSpec SGPT2 = undefined
      finalLayerNormSpec ST5 = NamedModel "final_layer_norm." $ layerNormSpec' SWithoutBias
      finalLayerNormSpec SByT5 = NamedModel "final_layer_norm." $ layerNormSpec' SWithoutBias
      finalLayerNormSpec SBART = ()
      finalLayerNormSpec SMBART = ()
      finalLayerNormSpec SPegasus = NamedModel "layer_norm." $ layerNormSpec' SWithBias
      finalLayerNormSpec SBERT = ()
      finalLayerNormSpec SRoBERTa = ()
      finalLayerNormSpec SGPT2 = undefined
      finalDropoutSpec ST5 = Dropout dropoutP
      finalDropoutSpec SByT5 = Dropout dropoutP
      finalDropoutSpec SBART = ()
      finalDropoutSpec SMBART = ()
      finalDropoutSpec SPegasus = ()
      finalDropoutSpec SBERT = ()
      finalDropoutSpec SRoBERTa = ()
      finalDropoutSpec SGPT2 = undefined
   in GTransformer (posEncSpec style) (relPosEncSpec style) (initialLayerNormSpec style) (initialDropoutSpec style) (stackSpec style) (finalLayerNormSpec style) (finalDropoutSpec style)
  where
    stackSpec' :: _
    stackSpec' style' = encoderStackSpec style' numLayers gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP eps
    layerNormSpec' :: _
    layerNormSpec' hasBias = LayerNormSpec hasBias gradient device dataType (SShape $ inputEmbedDim :|: SNil) eps
    relPosEncSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
    posEncSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim inputEmbedDim SNothing

-- | Specifies the absolute positional encoding layer of a transformer decoder.
type family
  TDPosEncF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDPosEncF 'T5 _ _ _ _ _ = ()
  TDPosEncF 'ByT5 gradient device dataType inputEmbedDim posEncDim = TDPosEncF 'T5 gradient device dataType inputEmbedDim posEncDim
  TDPosEncF 'BART gradient device dataType inputEmbedDim posEncDim = NamedModel (Embedding gradient ('Layout 'Dense) device dataType posEncDim inputEmbedDim 'Nothing)
  TDPosEncF 'MBART gradient device dataType inputEmbedDim posEncDim = TDPosEncF 'BART gradient device dataType inputEmbedDim posEncDim
  TDPosEncF 'Pegasus gradient device dataType inputEmbedDim posEncDim = TDPosEncF 'BART gradient device dataType inputEmbedDim posEncDim

-- | Specifies the relative positional encoding layer of a transformer decoder.
type family
  TDRelPosEncF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDRelPosEncF 'T5 gradient device dataType headDim posEncDim = NamedModel (Embedding gradient ('Layout 'Dense) device dataType posEncDim headDim 'Nothing)
  TDRelPosEncF 'ByT5 gradient device dataType headDim posEncDim = TDRelPosEncF 'T5 gradient device dataType headDim posEncDim
  TDRelPosEncF 'BART _ _ _ _ _ = ()
  TDRelPosEncF 'MBART gradient device dataType headDim posEncDim = TDRelPosEncF 'BART gradient device dataType headDim posEncDim
  TDRelPosEncF 'Pegasus gradient device dataType headDim posEncDim = TDRelPosEncF 'BART gradient device dataType headDim posEncDim

-- | Specifies the initial layer normalization layer of a transformer decoder.
type family
  TDInitialLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDInitialLayerNormF 'T5 _ _ _ _ = ()
  TDInitialLayerNormF 'ByT5 gradient device dataType inputEmbedDim = TDInitialLayerNormF 'T5 gradient device dataType inputEmbedDim
  TDInitialLayerNormF 'BART gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))
  TDInitialLayerNormF 'MBART gradient device dataType inputEmbedDim = TDInitialLayerNormF 'BART gradient device dataType inputEmbedDim
  TDInitialLayerNormF 'Pegasus _ _ _ _ = ()

-- | Specifies the initial dropout layer of a transformer decoder.
type family
  TDInitialDropoutF
    (style :: TransformerStyle) ::
    Type
  where
  TDInitialDropoutF 'T5 = Dropout
  TDInitialDropoutF 'ByT5 = Dropout
  TDInitialDropoutF 'BART = Dropout
  TDInitialDropoutF 'MBART = Dropout
  TDInitialDropoutF 'Pegasus = Dropout

-- | Specifies the transformer block stack of a transformer decoder.
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
    NamedModel
      ( GTransformerStack
          ( VS.Vector
              numLayers
              ( GTransformerBlock
                  (DecoderBlockSelfAttentionF style gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim)
                  (DecoderBlockCrossAttentionF style gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim)
                  (DecoderBlockFeedForwardNetworkF style gradient device dataType decoderInputEmbedDim ffnDim)
              )
          )
      )

-- | Specifies the final layer normalization layer of a transformer decoder.
type family
  TDFinalLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  TDFinalLayerNormF 'T5 gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[inputEmbedDim]))
  TDFinalLayerNormF 'ByT5 gradient device dataType inputEmbedDim = TDFinalLayerNormF 'T5 gradient device dataType inputEmbedDim
  TDFinalLayerNormF 'BART _ _ _ _ = ()
  TDFinalLayerNormF 'MBART gradient device dataType inputEmbedDim = TDFinalLayerNormF 'BART gradient device dataType inputEmbedDim
  TDFinalLayerNormF 'Pegasus gradient device dataType inputEmbedDim = NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[inputEmbedDim]))

-- | Specifies the final dropout layer of a transformer decoder.
type family
  TDFinalDropoutF
    (style :: TransformerStyle) ::
    Type
  where
  TDFinalDropoutF 'T5 = Dropout
  TDFinalDropoutF 'ByT5 = Dropout
  TDFinalDropoutF 'BART = ()
  TDFinalDropoutF 'MBART = ()
  TDFinalDropoutF 'Pegasus = ()

-- | Specifies the parameters of a transformer in a decoder configuration.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @decoderInputEmbedDim@: the dimension of the decoder input embeddings.
-- - @encoderOutputEmbedDim@: the dimension of the encoder output embeddings.
-- - @ffnDim@: the dimension of the feed-forward network.
-- - @posEncDim@: the dimension of the positional encoding.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
transformerDecoderSpec ::
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
  ModelSpec
    ( GTransformer
        (TDPosEncF style gradient device dataType decoderInputEmbedDim posEncDim)
        (TDRelPosEncF style gradient device dataType headDim posEncDim)
        (TDInitialLayerNormF style gradient device dataType decoderInputEmbedDim)
        (TDInitialDropoutF style)
        (TDStackF style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim)
        (TDFinalLayerNormF style gradient device dataType decoderInputEmbedDim)
        (TDFinalDropoutF style)
    )
transformerDecoderSpec style numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps =
  let posEncSpec ST5 = ()
      posEncSpec SByT5 = ()
      posEncSpec SBART = NamedModel "embed_positions." posEncSpec'
      posEncSpec SMBART = NamedModel "embed_positions." posEncSpec'
      posEncSpec SPegasus = NamedModel "embed_positions." posEncSpec'
      posEncSpec SBERT = undefined
      posEncSpec SRoBERTa = undefined
      posEncSpec SGPT2 = undefined
      relPosEncSpec ST5 = NamedModel "block.0.layer.0.SelfAttention.relative_attention_bias." relPosEncSpec'
      relPosEncSpec SByT5 = NamedModel "block.0.layer.0.SelfAttention.relative_attention_bias." relPosEncSpec'
      relPosEncSpec SBART = ()
      relPosEncSpec SMBART = ()
      relPosEncSpec SPegasus = ()
      relPosEncSpec SBERT = undefined
      relPosEncSpec SRoBERTa = undefined
      relPosEncSpec SGPT2 = undefined
      initialLayerNormSpec ST5 = ()
      initialLayerNormSpec SByT5 = ()
      initialLayerNormSpec SBART = NamedModel "layernorm_embedding." $ layerNormSpec' SWithBias
      initialLayerNormSpec SMBART = NamedModel "layernorm_embedding." $ layerNormSpec' SWithBias
      initialLayerNormSpec SPegasus = ()
      initialLayerNormSpec SBERT = undefined
      initialLayerNormSpec SRoBERTa = undefined
      initialLayerNormSpec SGPT2 = undefined
      initialDropoutSpec ST5 = Dropout dropoutP
      initialDropoutSpec SByT5 = Dropout dropoutP
      initialDropoutSpec SBART = Dropout dropoutP
      initialDropoutSpec SMBART = Dropout dropoutP
      initialDropoutSpec SPegasus = Dropout dropoutP
      initialDropoutSpec SBERT = undefined
      initialDropoutSpec SRoBERTa = undefined
      initialDropoutSpec SGPT2 = undefined
      stackSpec ST5 = NamedModel "block." $ stackSpec' ST5
      stackSpec SByT5 = NamedModel "block." $ stackSpec' SByT5
      stackSpec SBART = NamedModel "layers." $ stackSpec' SBART
      stackSpec SMBART = NamedModel "layers." $ stackSpec' SMBART
      stackSpec SPegasus = NamedModel "layers." $ stackSpec' SPegasus
      stackSpec SBERT = undefined
      stackSpec SRoBERTa = undefined
      stackSpec SGPT2 = undefined
      finalLayerNormSpec ST5 = NamedModel "final_layer_norm." $ layerNormSpec' SWithoutBias
      finalLayerNormSpec SByT5 = NamedModel "final_layer_norm." $ layerNormSpec' SWithoutBias
      finalLayerNormSpec SBART = ()
      finalLayerNormSpec SMBART = ()
      finalLayerNormSpec SPegasus = NamedModel "layer_norm." $ layerNormSpec' SWithBias
      finalLayerNormSpec SBERT = undefined
      finalLayerNormSpec SRoBERTa = undefined
      finalLayerNormSpec SGPT2 = undefined
      finalDropoutSpec ST5 = Dropout dropoutP
      finalDropoutSpec SByT5 = Dropout dropoutP
      finalDropoutSpec SBART = ()
      finalDropoutSpec SMBART = ()
      finalDropoutSpec SPegasus = ()
      finalDropoutSpec SBERT = undefined
      finalDropoutSpec SRoBERTa = undefined
      finalDropoutSpec SGPT2 = undefined
   in GTransformer (posEncSpec style) (relPosEncSpec style) (initialLayerNormSpec style) (initialDropoutSpec style) (stackSpec style) (finalLayerNormSpec style) (finalDropoutSpec style)
  where
    stackSpec' :: _
    stackSpec' style' = decoderStackSpec style' numLayers gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP eps
    layerNormSpec' :: _
    layerNormSpec' hasBias = LayerNormSpec hasBias gradient device dataType (SShape $ decoderInputEmbedDim :|: SNil) eps
    relPosEncSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim headDim SNothing
    posEncSpec' = EmbeddingSpec gradient (SLayout SDense) device dataType posEncDim decoderInputEmbedDim SNothing

instance
  ( HasInitialize posEnc generatorDevice posEnc generatorDevice,
    HasInitialize relPosEnc generatorDevice relPosEnc generatorDevice,
    HasInitialize initialLayerNorm generatorDevice initialLayerNorm generatorDevice,
    HasInitialize initialDropout generatorDevice initialDropout generatorDevice,
    HasInitialize stack generatorDevice stack generatorDevice,
    HasInitialize finalLayerNorm generatorDevice finalLayerNorm generatorDevice,
    HasInitialize finalDropout generatorDevice finalDropout generatorDevice
  ) =>
  HasInitialize
    (GTransformer posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    generatorDevice
    (GTransformer posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    generatorDevice
  where
  initialize (GTransformer posEncSpec relPosEncSpec initialLayerNormSpec initialDropoutSpec stackSpec finalLayerNormSpec finalDropoutSpec) =
    let posEnc = IxStateT . initialize $ posEncSpec
        relPosEnc = IxStateT . initialize $ relPosEncSpec
        initialLayerNorm = IxStateT . initialize $ initialLayerNormSpec
        initialDropout = IxStateT . initialize $ initialDropoutSpec
        stack = IxStateT . initialize $ stackSpec
        finalLayerNorm = IxStateT . initialize $ finalLayerNormSpec
        finalDropout = IxStateT . initialize $ finalDropoutSpec
     in runIxStateT $
          GTransformer
            <<$>> posEnc
            <<*>> relPosEnc
            <<*>> initialLayerNorm
            <<*>> initialDropout
            <<*>> stack
            <<*>> finalLayerNorm
            <<*>> finalDropout

instance
  ( HasStateDict posEnc,
    HasStateDict relPosEnc,
    HasStateDict initialLayerNorm,
    HasStateDict initialDropout,
    HasStateDict stack,
    HasStateDict finalLayerNorm,
    HasStateDict finalDropout
  ) =>
  HasStateDict (GTransformer posEnc relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
  where
  fromStateDict (GTransformer posEncSpec relPosEncSpec initialLayerNormSpec initialDropoutSpec stackSpec finalLayerNormSpec finalDropoutSpec) k =
    GTransformer
      <$> fromStateDict posEncSpec k
      <*> fromStateDict relPosEncSpec k
      <*> fromStateDict initialLayerNormSpec k
      <*> fromStateDict initialDropoutSpec k
      <*> fromStateDict stackSpec k
      <*> fromStateDict finalLayerNormSpec k
      <*> fromStateDict finalDropoutSpec k
  toStateDict k GTransformer {..} = do
    () <- toStateDict k tPosEnc
    () <- toStateDict k tRelPosEnc
    () <- toStateDict k tInitialLayerNorm
    () <- toStateDict k tInitialDropout
    () <- toStateDict k tStack
    () <- toStateDict k tFinalLayerNorm
    () <- toStateDict k tFinalDropout
    pure ()

-- | 'HasForward' instance for 'GTransformer' in an encoder configuration
-- with absolute positional encoding rather than relative positional encoding.
--
-- @
-- ┌───────┐  ┌─────┐  ┌───────────────┐
-- │ input │  │ pos │  │ attentionMask │
-- └───┬───┘  └─────┘  └───────┬───────┘
--     │         │             │
--     │         ▼             │
--     │      tPosEnc          │
--     │         │             │
--     └──►add◄──┘             │
--          │                  │
--          ▼                  │
-- (tInitialLayerNorm)         │
--          ▼                  ▼
--  (tInitialDropout)     unsqueeze
--          ▼                  │
--       tStack◄───────────────┘
--          ▼
--  (tFinalLayerNorm)
--          ▼
--   (tFinalDropout)
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      posEnc
      (Tensor posGradient posLayout posDevice posDataType posShape)
      generatorDevice
      (Tensor posEncGradient posEncLayout posEncDevice posEncDataType posEncShape)
      generatorDevice0,
    HasForward
      initialLayerNorm
      ( Tensor
          (inputGradient <|> posEncGradient)
          (inputLayout <+> posEncLayout)
          (inputDevice <+> posEncDevice)
          (inputDataType <+> posEncDataType)
          (BroadcastShapesF inputShape posEncShape)
      )
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      initialDropout
      tensor1
      generatorDevice1
      tensor2
      generatorDevice2,
    HasForward
      stack
      ( tensor2,
        Tensor
          attentionMaskGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
      )
      generatorDevice2
      tensor3
      generatorDevice3,
    HasForward
      finalLayerNorm
      tensor3
      generatorDevice3
      tensor4
      generatorDevice4,
    HasForward
      finalDropout
      tensor4
      generatorDevice4
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformer posEnc () initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor posGradient posLayout posDevice posDataType posShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformer {..} (input, pos, attentionMask) =
    let attentionBias = unsqueeze @('SelectDim ('ByIndex 1)) attentionMask
     in runIxStateT $
          ireturn pos
            >>>= IxStateT . forward tPosEnc
            >>>= ireturn . (input `add`)
            >>>= IxStateT . forward tInitialLayerNorm
            >>>= IxStateT . forward tInitialDropout
            >>>= (\input' -> IxStateT $ forward tStack (input', attentionBias))
            >>>= IxStateT . forward tFinalLayerNorm
            >>>= IxStateT . forward tFinalDropout

-- | 'HasForward' instance for 'GTransformer' in an encoder configuration
-- with relative positional encoding rather than absolute positional encoding.
--
-- @
--      ┌───────┐  ┌────────┐  ┌───────────────┐
--      │ input │  │ relPos │  │ attentionMask │
--      └───┬───┘  └───┬────┘  └───────┬───────┘
--          │          │               │
--          │          ▼               │
--          │     tRelPosEnc           │
--          │          ▼               │
--          │      transpose           │
--          │          ▼               ▼
--          │      transpose       unsqueeze
--          ▼          │               │
-- (tInitialLayerNorm) │               │
--          ▼          └─────►add◄─────┘
--  (tInitialDropout)          │
--          ▼                  │
--       tStack◄───────────────┘
--          ▼
--  (tFinalLayerNorm)
--          ▼
--   (tFinalDropout)
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      initialLayerNorm
      (Tensor inputGradient inputLayout inputDevice inputDataType inputShape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      initialDropout
      tensor0
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      relPosEnc
      (Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape)
      generatorDevice1
      (Tensor relPosEncGradient relPosEncLayout relPosEncDevice relPosEncDataType relPosEncShape)
      generatorDevice2,
    HasForward
      stack
      ( tensor1,
        Tensor
          (relPosEncGradient <|> attentionMaskGradient)
          (relPosEncLayout <+> attentionMaskLayout)
          (relPosEncDevice <+> attentionMaskDevice)
          (relPosEncDataType <+> attentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      relPosEncShape
                  )
              )
              (UnsqueezeF ('SelectDim ('ByIndex 1)) attentionMaskShape)
          )
      )
      generatorDevice2
      tensor3
      generatorDevice3,
    HasForward
      finalLayerNorm
      tensor3
      generatorDevice3
      tensor4
      generatorDevice4,
    HasForward
      finalDropout
      tensor4
      generatorDevice4
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformer () relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    ( Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
      Tensor relPosGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformer {..} (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxStateT . forward tRelPosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) attentionMask)
     in runIxStateT $
          ireturn input
            >>>= IxStateT . forward tInitialLayerNorm
            >>>= IxStateT . forward tInitialDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxStateT $ forward tStack (input', attentionBias')))
            >>>= IxStateT . forward tFinalLayerNorm
            >>>= IxStateT . forward tFinalDropout

-- | 'HasForward' instance for 'GTransformer' in a decoder configuration
-- with absolute positional encoding rather than relative positional encoding.
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
--        (tInitialLayerNorm)                │                     │                         │
--                 ▼                         │                     ▼                         ▼
--         (tInitialDropout)                 │                 unsqueeze                 unsqueeze
--                 ▼                         │                     │                         │
--              tStack◄──────────────────────┘◄────────────────────┘◄────────────────────────┘
--                 ▼
--         (tFinalLayerNorm)
--                 ▼
--          (tFinalDropout)
--                 │
--                 ▼
--            ┌────────┐
--            │ output │
--            └────────┘
-- @
instance
  ( HasForward
      posEnc
      (Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape)
      generatorDevice
      (Tensor decoderPosEncGradient decoderPosEncLayout decoderPosEncDevice decoderPosEncDataType decoderPosEncShape)
      generatorDevice0,
    HasForward
      initialLayerNorm
      ( Tensor
          (decoderInputGradient <|> decoderPosEncGradient)
          (decoderInputLayout <+> decoderPosEncLayout)
          (decoderInputDevice <+> decoderPosEncDevice)
          (decoderInputDataType <+> decoderPosEncDataType)
          (BroadcastShapesF decoderInputShape decoderPosEncShape)
      )
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      initialDropout
      tensor1
      generatorDevice1
      tensor2
      generatorDevice2,
    HasForward
      stack
      ( tensor2,
        Tensor encoderOutputGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape,
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
      generatorDevice2
      tensor3
      generatorDevice3,
    HasForward
      finalLayerNorm
      tensor3
      generatorDevice3
      tensor4
      generatorDevice4,
    HasForward
      finalDropout
      tensor4
      generatorDevice4
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformer posEnc () initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor encoderOutputGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape,
      Tensor decoderPosGradient decoderPosLayout decoderPosDevice decoderPosDataType decoderPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformer {..} (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) =
    let decoderAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderPos
            >>>= IxStateT . forward tPosEnc
            >>>= ireturn . (decoderInput `add`)
            >>>= IxStateT . forward tInitialLayerNorm
            >>>= IxStateT . forward tInitialDropout
            >>>= ( \decoderInput' ->
                     IxStateT $
                       forward
                         tStack
                         ( decoderInput',
                           encoderOutput,
                           decoderAttentionBias,
                           crossAttentionBias
                         )
                 )
            >>>= IxStateT . forward tFinalLayerNorm
            >>>= IxStateT . forward tFinalDropout

-- | 'HasForward' instance for 'GTransformer' in a decoder configuration
-- with relative positional encoding rather than absolute positional encoding.
--
-- @
--   ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐  ┌────────────────────┐
--   │ decoderInput │  │ encoderOutput │  │ decoderRelPos │  │ decoderAttentionMask │  │ crossAttentionMask │
--   └──────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  └─────────┬──────────┘
--          │                  │                  │                     │                        │
--          │                  │                  ▼                     │                        │
--          │                  │             tdRelPosEnc                │                        │
--          │                  │                  ▼                     │                        │
--          │                  │              transpose                 │                        │
--          │                  │                  ▼                     ▼                        ▼
--          │                  │              transpose             unsqueeze                unsqueeze
--          ▼                  │                  │                     │                        │
-- (tInitialLayerNorm)         │                  │                     │                        │
--          ▼                  │                  └────────►add◄────────┘                        │
--  (tInitialDropout)          │                             │                                   │
--          ▼                  │                             │                                   │
--       tStack◄───────────────┘◄────────────────────────────┘◄──────────────────────────────────┘
--          ▼
--  (tFinalLayerNorm)
--          ▼
--   (tFinalDropout)
--          │
--          ▼
--     ┌────────┐
--     │ output │
--     └────────┘
-- @
instance
  ( HasForward
      initialLayerNorm
      (Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      initialDropout
      tensor0
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      relPosEnc
      (Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape)
      generatorDevice1
      (Tensor decoderRelPosEncGradient decoderRelPosEncLayout decoderRelPosEncDevice decoderRelPosEncDataType decoderRelPosEncShape)
      generatorDevice2,
    HasForward
      stack
      ( tensor1,
        Tensor encoderOutputGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape,
        Tensor
          (decoderRelPosEncGradient <|> decoderAttentionMaskGradient)
          (decoderRelPosEncLayout <+> decoderAttentionMaskLayout)
          (decoderRelPosEncDevice <+> decoderAttentionMaskDevice)
          (decoderRelPosEncDataType <+> decoderAttentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      decoderRelPosEncShape
                  )
              )
              (UnsqueezeF ('SelectDim ('ByIndex 1)) decoderAttentionMaskShape)
          ),
        Tensor
          crossAttentionMaskGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          (UnsqueezeF ('SelectDim ('ByIndex 1)) crossAttentionMaskShape)
      )
      generatorDevice2
      tensor3
      generatorDevice3,
    HasForward
      finalLayerNorm
      tensor3
      generatorDevice3
      tensor4
      generatorDevice4,
    HasForward
      finalDropout
      tensor4
      generatorDevice4
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GTransformer () relPosEnc initialLayerNorm initialDropout stack finalLayerNorm finalDropout)
    ( Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor encoderOutputGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape,
      Tensor decoderRelPosGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GTransformer {..} (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxStateT . forward tRelPosEnc
            >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @('SelectDim ('ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @('SelectDim ('ByIndex 1)) crossAttentionMask
     in runIxStateT $
          ireturn decoderInput
            >>>= IxStateT . forward tInitialLayerNorm
            >>>= IxStateT . forward tInitialDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxStateT $
                                  forward
                                    tStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxStateT . forward tFinalLayerNorm
            >>>= IxStateT . forward tFinalDropout

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
      dropoutP = 0
      eps = 1e-6
  let g = sMkGenerator device 0
      spec = NamedModel "encoder." $ transformerEncoderSpec ST5 (SNat @10) gradient device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP eps
  (encoder, g') <- initialize spec g
  encoder' <- flip evalStateT Map.empty $ do
    toStateDict mempty encoder
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' dataType (SShape $ batchDim :|: seqDim :|: inputEmbedDim :|: SNil)
      relPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      attentionMask = sOnes' dataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward encoder' (input, relPos, attentionMask) g'
  pure output

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
      dropoutP = 0
      eps = 1e-6
  let g = sMkGenerator device 0
      spec = NamedModel "decoder." $ transformerDecoderSpec SBART (SNat @10) gradient device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim posEncDim dropoutP eps
  (decoder, g') <- initialize spec g
  decoder' <- flip evalStateT Map.empty $ do
    toStateDict mempty decoder
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      decoderInput = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderInputEmbedDim :|: SNil)
      encoderOutput = sOnes' dataType (SShape $ batchDim :|: seqDim :|: encoderOutputEmbedDim :|: SNil)
      decoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' dataType (SShape $ batchDim :|: decoderSeqDim :|: seqDim :|: SNil)
  (output, _) <- forward decoder' (decoderInput, encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask) g'
  pure output
