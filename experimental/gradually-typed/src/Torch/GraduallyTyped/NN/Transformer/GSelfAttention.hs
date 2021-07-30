{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.GSelfAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.GMultiHeadAttention (DropoutF, GMultiHeadAttention, KInProjF, OutProjF, QInProjF, VInProjF, multiHeadAttentionSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic self-attention layer data type.
--
-- - @initialLayerNorm@: the initial layer normalization
-- - @mha@: the multi-headed attention layer
-- - @dropout@: the dropout layer
-- - @finalLayerNorm@: the final layer normalization
data
  GSelfAttention
    (initialLayerNorm :: Type)
    (mha :: Type)
    (dropout :: Type)
    (finalLayerNorm :: Type)
  where
  GSelfAttention ::
    forall initialLayerNorm mha dropout finalLayerNorm.
    { -- | initial layer normalization of the self-attention layer.
      saInitialLayerNorm :: initialLayerNorm,
      -- | multi-headed attention layer specialized for self-attention.
      saMultiHeadAttention :: mha,
      -- | dropout
      saDropout :: dropout,
      -- | final layer normalization of the self-attention layer.
      saFinalLayerNorm :: finalLayerNorm
    } ->
    GSelfAttention initialLayerNorm mha dropout finalLayerNorm

type instance
  ModelSpec (GSelfAttention initialLayerNorm mha dropout finalLayerNorm) =
    GSelfAttention (ModelSpec initialLayerNorm) (ModelSpec mha) (ModelSpec dropout) (ModelSpec finalLayerNorm)

-- | Specifies the initial layer normalization of the self-attention layer.
type family
  SAInitialLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim]))
  SAInitialLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    SAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim
  SAInitialLayerNormF 'BART _ _ _ _ =
    ()
  SAInitialLayerNormF 'MBART gradient device dataType queryEmbedDim =
    SAInitialLayerNormF 'BART gradient device dataType queryEmbedDim
  SAInitialLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  SAInitialLayerNormF 'BERT _ _ _ _ =
    ()
  SAInitialLayerNormF 'RoBERTa gradient device dataType queryEmbedDim =
    SAInitialLayerNormF 'BERT gradient device dataType queryEmbedDim

-- | Specifies the multi-headed attention layer of the self-attention layer.
type family
  SAMultiheadAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim =
    NamedModel
      ( GMultiHeadAttention
          headDim
          headEmbedDim
          embedDim
          (QInProjF style gradient device dataType queryEmbedDim embedDim)
          (KInProjF style gradient device dataType queryEmbedDim embedDim)
          (VInProjF style gradient device dataType queryEmbedDim embedDim)
          (OutProjF style gradient device dataType embedDim queryEmbedDim)
          (DropoutF style)
      )

-- | Specifies the dropout layer of the self-attention layer.
type family
  SADropoutF
    (style :: TransformerStyle) ::
    Type
  where
  SADropoutF _ = Dropout

-- | Specifies the final layer normalization of the self-attention layer.
type family
  SAFinalLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SAFinalLayerNormF 'T5 _ _ _ _ =
    ()
  SAFinalLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    SAFinalLayerNormF 'T5 gradient device dataType queryEmbedDim
  SAFinalLayerNormF 'BART gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  SAFinalLayerNormF 'MBART gradient device dataType queryEmbedDim =
    SAFinalLayerNormF 'BART gradient device dataType queryEmbedDim
  SAFinalLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    ()
  SAFinalLayerNormF 'BERT gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  SAFinalLayerNormF 'RoBERTa gradient device dataType queryEmbedDim =
    SAFinalLayerNormF 'BERT gradient device dataType queryEmbedDim

-- | Specifies the parameters of a self-attention layer.
--
-- - @style@: the style of the transformer stack, e.g. 'ST5', 'SByT5', etc.
-- - @gradient@: whether to compute the gradient of the stack's parameters.
-- - @device@: the computational device on which the stack is allocated.
-- - @dataType@: the data type of the stack's parameters.
-- - @headDim@: the dimension of all transformer heads in the stack.
-- - @headEmbedDim@: the dimension of the transformer head embeddings.
-- - @embedDim@: the dimension of the transformer embeddings.
-- - @queryEmbedDim@: the dimension of the transformer query embeddings.
-- - @dropoutP@: the dropout rate.
-- - @eps@: the epsilon value for numerical stability of the layer normalization.
selfAttentionSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  Double ->
  Double ->
  ModelSpec
    ( GSelfAttention
        (SAInitialLayerNormF style gradient device dataType queryEmbedDim)
        (SAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim)
        (SADropoutF style)
        (SAFinalLayerNormF style gradient device dataType queryEmbedDim)
    )
selfAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps =
  let initialLayerNormSpec ST5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SByT5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SBART = ()
      initialLayerNormSpec SMBART = ()
      initialLayerNormSpec SPegasus = NamedModel "self_attn_layer_norm." layerNormWithBiasSpec
      initialLayerNormSpec SBERT = ()
      initialLayerNormSpec SRoBERTa = ()
      initialLayerNormSpec SGPT2 = undefined
      mhaSpec ST5 = NamedModel "SelfAttention." $ mhaSpec' ST5
      mhaSpec SByT5 = NamedModel "SelfAttention." $ mhaSpec' SByT5
      mhaSpec SBART = NamedModel "self_attn." $ mhaSpec' SBART
      mhaSpec SMBART = NamedModel "self_attn." $ mhaSpec' SMBART
      mhaSpec SPegasus = NamedModel "self_attn." $ mhaSpec' SPegasus
      mhaSpec SBERT = NamedModel mempty $ mhaSpec' SBERT
      mhaSpec SRoBERTa = NamedModel mempty $ mhaSpec' SRoBERTa
      mhaSpec SGPT2 = undefined
      dropoutSpec _ = Dropout dropoutP
      finalLayerNormSpec ST5 = ()
      finalLayerNormSpec SByT5 = ()
      finalLayerNormSpec SBART = NamedModel "self_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SMBART = NamedModel "self_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SPegasus = ()
      finalLayerNormSpec SBERT = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      finalLayerNormSpec SRoBERTa = NamedModel "output.LayerNorm." layerNormWithBiasSpec
      finalLayerNormSpec SGPT2 = undefined
   in GSelfAttention (initialLayerNormSpec style) (mhaSpec style) (dropoutSpec style) (finalLayerNormSpec style)
  where
    mhaSpec' ::
      STransformerStyle style ->
      ModelSpec
        ( GMultiHeadAttention
            headDim
            headEmbedDim
            embedDim
            (QInProjF style gradient device dataType queryEmbedDim embedDim)
            (KInProjF style gradient device dataType queryEmbedDim embedDim)
            (VInProjF style gradient device dataType queryEmbedDim embedDim)
            (OutProjF style gradient device dataType embedDim queryEmbedDim)
            (DropoutF style)
        )
    mhaSpec' style' = multiHeadAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP
    layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
    layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps

instance
  ( HasInitialize initialLayerNorm generatorDevice initialLayerNorm generatorDevice,
    HasInitialize multiHeadAttention generatorDevice multiHeadAttention generatorDevice,
    HasInitialize dropout generatorDevice dropout generatorDevice,
    HasInitialize finalLayerNorm generatorDevice finalLayerNorm generatorDevice
  ) =>
  HasInitialize
    (GSelfAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    generatorDevice
    (GSelfAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    generatorDevice
  where
  initialize (GSelfAttention initialLayerNormSpec mhaSpec dropoutSpec finalLayerNormSpec) =
    let initialLayerNorm = IxStateT . initialize $ initialLayerNormSpec
        multiHeadAttention = IxStateT . initialize $ mhaSpec
        dropout = IxStateT . initialize $ dropoutSpec
        finalLayerNorm = IxStateT . initialize $ finalLayerNormSpec
     in runIxStateT (GSelfAttention <<$>> initialLayerNorm <<*>> multiHeadAttention <<*>> dropout <<*>> finalLayerNorm)

instance
  ( HasStateDict initialLayerNorm,
    HasStateDict multiHeadAttention,
    HasStateDict dropout,
    HasStateDict finalLayerNorm
  ) =>
  HasStateDict (GSelfAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
  where
  fromStateDict (GSelfAttention initialLayerNormSpec mhaSpec dropoutSpec finalLayerNormSpec) k =
    GSelfAttention
      <$> fromStateDict initialLayerNormSpec k
      <*> fromStateDict mhaSpec k
      <*> fromStateDict dropoutSpec k
      <*> fromStateDict finalLayerNormSpec k
  toStateDict k GSelfAttention {..} = do
    () <- toStateDict k saInitialLayerNorm
    () <- toStateDict k saMultiHeadAttention
    () <- toStateDict k saDropout
    () <- toStateDict k saFinalLayerNorm
    pure ()

-- | 'HasForward' instance for 'GSelfAttention'.
--
-- @
-- ┌───────────────┐     ┌───────┐
-- │ attentionBias │     │ query │
-- └───────┬───────┘     └───┬───┘
--         │                 │
--         │           ┌─────┴─────┐
--         │           │           │
--         │           ▼           │
--         │  (saInitialLayerNorm) │
--         │           │           │
--         │      ┌────┼────┐      │
--         │      │    │    │      │
--         │      ▼    ▼    ▼      │
--         └─►saMultiHeadAttention │
--                     │           │
--                     ▼           │
--                 saDropout       │
--                     │           │
--                     └───►add◄───┘
--                           │
--                           ▼
--                   (saFinalLayerNorm)
--                           │
--                           ▼
--                       ┌───────┐
--                       │ query │
--                       └───────┘
-- @
instance
  ( HasForward
      initialLayerNorm
      (Tensor queryGradient queryLayout queryDevice queryDataType queryShape)
      generatorDevice
      tensor0
      generatorDevice0,
    HasForward
      multiHeadAttention
      ( tensor0,
        tensor0,
        tensor0,
        Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      generatorDevice0
      tensor1
      generatorDevice1,
    HasForward
      dropout
      tensor1
      generatorDevice1
      (Tensor gradient2 layout2 device2 dataType2 shape2)
      generatorDevice2,
    HasForward
      finalLayerNorm
      (Tensor (queryGradient <|> gradient2) (queryLayout <+> layout2) (queryDevice <+> device2) (queryDataType <+> dataType2) (BroadcastShapesF queryShape shape2))
      generatorDevice2
      output
      generatorOutputDevice
  ) =>
  HasForward
    (GSelfAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    ( Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GSelfAttention {..} (query, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward saInitialLayerNorm
        >>>= (\query' -> IxStateT $ forward saMultiHeadAttention (query', query', query', attentionBias))
        >>>= IxStateT . forward saDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward saFinalLayerNorm
