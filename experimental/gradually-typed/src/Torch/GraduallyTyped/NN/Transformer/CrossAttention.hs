{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.CrossAttention where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map as Map
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec, NamedModel (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (DropoutF, GMultiHeadAttention, KInProjF, OutProjF, QInProjF, VInProjF, multiHeadAttentionSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic cross-attention layer data type.
--
-- - @initialLayerNorm@: the initial layer normalization
-- - @mha@: the multi-headed attention layer
-- - @dropout@: the dropout layer
-- - @finalLayerNorm@: the final layer normalization
data
  GCrossAttention
    (initialLayerNorm :: Type)
    (mha :: Type)
    (dropout :: Type)
    (finalLayerNorm :: Type)
  where
  GCrossAttention ::
    forall initialLayerNorm mha dropout finalLayerNorm.
    { -- | initial layer normalization of the cross-attention layer.
      caInitialLayerNorm :: initialLayerNorm,
      -- | multi-headed attention layer specialized for cross-attention.
      caMultiHeadAttention :: mha,
      -- | dropout
      caDropout :: dropout,
      -- | final layer normalization of the cross-attention layer.
      caFinalLayerNorm :: finalLayerNorm
    } ->
    GCrossAttention initialLayerNorm mha dropout finalLayerNorm

type instance
  ModelSpec (GCrossAttention initialLayerNorm mha dropout finalLayerNorm) =
    GCrossAttention (ModelSpec initialLayerNorm) (ModelSpec mha) (ModelSpec dropout) (ModelSpec finalLayerNorm)

type family
  CAInitialLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim]))
  CAInitialLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    CAInitialLayerNormF 'T5 gradient device dataType queryEmbedDim
  CAInitialLayerNormF 'BART _ _ _ _ =
    ()
  CAInitialLayerNormF 'MBART gradient device dataType queryEmbedDim =
    CAInitialLayerNormF 'BART gradient device dataType queryEmbedDim
  CAInitialLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    CAInitialLayerNormF 'BART gradient device dataType queryEmbedDim

type family
  CAMultiheadAttentionF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim =
    NamedModel
      ( GMultiHeadAttention
          headDim
          headEmbedDim
          embedDim
          (QInProjF style gradient device dataType queryEmbedDim embedDim)
          (KInProjF style gradient device dataType keyEmbedDim embedDim)
          (VInProjF style gradient device dataType keyEmbedDim embedDim)
          (OutProjF style gradient device dataType embedDim queryEmbedDim)
          (DropoutF style)
      )

type family
  CADropoutF
    (style :: TransformerStyle) ::
    Type
  where
  CADropoutF _ = Dropout

type family
  CAFinalLayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CAFinalLayerNormF 'T5 _ _ _ _ =
    ()
  CAFinalLayerNormF 'ByT5 gradient device dataType queryEmbedDim =
    CAFinalLayerNormF 'T5 gradient device dataType queryEmbedDim
  CAFinalLayerNormF 'BART gradient device dataType queryEmbedDim =
    NamedModel (LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim]))
  CAFinalLayerNormF 'MBART gradient device dataType queryEmbedDim =
    CAFinalLayerNormF 'BART gradient device dataType queryEmbedDim
  CAFinalLayerNormF 'Pegasus gradient device dataType queryEmbedDim =
    CAFinalLayerNormF 'BART gradient device dataType queryEmbedDim

crossAttentionSpec ::
  forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim.
  STransformerStyle style ->
  SGradient gradient ->
  SDevice device ->
  SDataType dataType ->
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim queryEmbedDim ->
  SDim keyEmbedDim ->
  Double ->
  Double ->
  ModelSpec
    ( GCrossAttention
        (CAInitialLayerNormF style gradient device dataType queryEmbedDim)
        (CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
        (CADropoutF style)
        (CAFinalLayerNormF style gradient device dataType queryEmbedDim)
    )
crossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps =
  let initialLayerNormSpec ST5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SByT5 = NamedModel "layer_norm." layerNormWithoutBiasSpec
      initialLayerNormSpec SBART = ()
      initialLayerNormSpec SMBART = ()
      initialLayerNormSpec SPegasus = ()
      initialLayerNormSpec SBERT = undefined
      initialLayerNormSpec SRoBERTa = undefined
      initialLayerNormSpec SGPT2 = undefined
      mhaSpec ST5 = NamedModel "EncDecAttention." $ mhaSpec' ST5
      mhaSpec SByT5 = NamedModel "EncDecAttention." $ mhaSpec' SByT5
      mhaSpec SBART = NamedModel "encoder_attn." $ mhaSpec' SBART
      mhaSpec SMBART = NamedModel "encoder_attn." $ mhaSpec' SMBART
      mhaSpec SPegasus = NamedModel "encoder_attn." $ mhaSpec' SPegasus
      mhaSpec SBERT = undefined
      mhaSpec SRoBERTa = undefined
      mhaSpec SGPT2 = undefined
      dropoutSpec _ = Dropout dropoutP
      finalLayerNormSpec ST5 = ()
      finalLayerNormSpec SByT5 = ()
      finalLayerNormSpec SBART = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SMBART = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SPegasus = NamedModel "encoder_attn_layer_norm." layerNormWithBiasSpec
      finalLayerNormSpec SBERT = undefined
      finalLayerNormSpec SRoBERTa = undefined
      finalLayerNormSpec SGPT2 = undefined
   in GCrossAttention (initialLayerNormSpec style) (mhaSpec style) (dropoutSpec style) (finalLayerNormSpec style)
  where
    mhaSpec' ::
      STransformerStyle style ->
      ModelSpec
        ( GMultiHeadAttention
            headDim
            headEmbedDim
            embedDim
            (QInProjF style gradient device dataType queryEmbedDim embedDim)
            (KInProjF style gradient device dataType keyEmbedDim embedDim)
            (VInProjF style gradient device dataType keyEmbedDim embedDim)
            (OutProjF style gradient device dataType embedDim queryEmbedDim)
            (DropoutF style)
        )
    mhaSpec' style' = multiHeadAttentionSpec style' gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP
    layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
    layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps

instance
  ( HasInitialize initialLayerNorm generatorDevice initialLayerNorm generatorDevice,
    HasInitialize multiHeadAttention generatorDevice multiHeadAttention generatorDevice,
    HasInitialize dropout generatorDevice dropout generatorDevice,
    HasInitialize finalLayerNorm generatorDevice finalLayerNorm generatorDevice
  ) =>
  HasInitialize
    (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    generatorDevice
    (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    generatorDevice
  where
  initialize (GCrossAttention initialLayerNormSpec mhaSpec dropoutSpec finalLayerNormSpec) =
    let initialLayerNorm = IxStateT . initialize $ initialLayerNormSpec
        multiHeadAttention = IxStateT . initialize $ mhaSpec
        dropout = IxStateT . initialize $ dropoutSpec
        finalLayerNorm = IxStateT . initialize $ finalLayerNormSpec
     in runIxStateT (GCrossAttention <<$>> initialLayerNorm <<*>> multiHeadAttention <<*>> dropout <<*>> finalLayerNorm)

instance
  ( HasStateDict initialLayerNorm,
    HasStateDict multiHeadAttention,
    HasStateDict dropout,
    HasStateDict finalLayerNorm
  ) =>
  HasStateDict (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
  where
  fromStateDict (GCrossAttention initialLayerNormSpec mhaSpec dropoutSpec finalLayerNormSpec) k =
    GCrossAttention
      <$> fromStateDict initialLayerNormSpec k
      <*> fromStateDict mhaSpec k
      <*> fromStateDict dropoutSpec k
      <*> fromStateDict finalLayerNormSpec k
  toStateDict k GCrossAttention {..} = do
    () <- toStateDict k caInitialLayerNorm
    () <- toStateDict k caMultiHeadAttention
    () <- toStateDict k caDropout
    () <- toStateDict k caFinalLayerNorm
    pure ()

-- | 'HasForward' instance for 'GCrossAttention'.
--
-- @
--        ┌───────┐    ┌─────┐    ┌───────────────┐
--        │ query │    │ key │    │ attentionBias │
--        └───┬───┘    └──┬──┘    └───────┬───────┘
--            │           │               │
-- ┌──────────┤           │               │
-- │          │           │               │
-- │          ▼           │               │
-- │ (caInitialLayerNorm) │               │
-- │          │           │               │
-- │          │       ┌───┴───┐           │
-- │          │       │       │           │
-- │          ▼       ▼       ▼           │
-- │        caMultiheadAttention◄─────────┘
-- │                  │
-- │                  ▼
-- │              caDropout
-- │                  │
-- └──────►add◄───────┘
--          │
--          ▼
--  (caFinalLayerNorm)
--          │
--          ▼
--      ┌───────┐
--      │ query │
--      └───────┘
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
        Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
        Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
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
    (GCrossAttention initialLayerNorm multiHeadAttention dropout finalLayerNorm)
    ( Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
      Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    generatorDevice
    output
    generatorOutputDevice
  where
  forward GCrossAttention {..} (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward caInitialLayerNorm
        >>>= (\query' -> IxStateT $ forward caMultiHeadAttention (query', key, key, attentionBias))
        >>>= IxStateT . forward caDropout
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward caFinalLayerNorm

testCA :: IO _
testCA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      dropoutP = 0
      eps = 1e-6
  let g = sMkGenerator device 0
      spec = NamedModel "ca." $ crossAttentionSpec SPegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps
  (ca, g') <- initialize spec g
  ca' <- flip evalStateT Map.empty $ do
    toStateDict mempty ca
    fromStateDict spec mempty
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @4
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward ca' (query, key, attentionBias) g'
  pure output
