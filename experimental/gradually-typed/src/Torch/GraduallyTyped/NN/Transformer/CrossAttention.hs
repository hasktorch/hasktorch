{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.OrIdempotenceL3C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.CrossAttention where

import Control.Monad.Indexed (IxPointed (ireturn), (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map as Map
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType, SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF)
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..), LayerNormSpec (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention, MultiHeadAttentionSpec (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..), SHasBias (..))
import Torch.GraduallyTyped.Random (sGeneratorToDevice, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))

-- | Generic cross-attention layer.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'CrossAttention'.
data
  GCrossAttention
    (mha :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GCrossAttention ::
    forall mha layerNorm dropout.
    { -- | cross-attention
      caMultiHeadAttention :: mha,
      -- | layer norm
      caLayerNorm :: layerNorm,
      -- | dropout
      caDropout :: dropout
    } ->
    GCrossAttention mha layerNorm dropout

-- | Cross-attention layer.
newtype
  CrossAttention
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  where
  CrossAttention ::
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim.
    GCrossAttention
      (CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
      (CALayerNormF style gradient device dataType queryEmbedDim)
      (CADropoutF style) ->
    CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim

data
  CrossAttentionSpec
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  where
  CrossAttentionSpec ::
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
    CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim

type instance ModelSpec (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim) = CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim

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
    MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim

type family
  CALayerNormF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CALayerNormF 'T5 gradient device dataType queryEmbedDim = LayerNorm 'WithoutBias gradient device dataType ('Shape '[queryEmbedDim])
  CALayerNormF 'ByT5 gradient device dataType queryEmbedDim = CALayerNormF 'T5 gradient device dataType queryEmbedDim
  CALayerNormF 'BART gradient device dataType queryEmbedDim = LayerNorm 'WithBias gradient device dataType ('Shape '[queryEmbedDim])
  CALayerNormF 'MBART gradient device dataType queryEmbedDim = CALayerNormF 'BART gradient device dataType queryEmbedDim
  CALayerNormF 'Pegasus gradient device dataType queryEmbedDim = CALayerNormF 'BART gradient device dataType queryEmbedDim

type family
  CADropoutF
    (style :: TransformerStyle) ::
    Type
  where
  CADropoutF _ = Dropout

instance
  ( multiHeadAttention ~ CAMultiheadAttentionF style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim,
    HasInitialize multiHeadAttention device multiHeadAttention device,
    layerNorm ~ CALayerNormF style gradient device dataType queryEmbedDim,
    HasInitialize layerNorm device layerNorm device,
    dropout ~ CADropoutF style
  ) =>
  HasInitialize
    (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    generatorDevice
    (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    device
  where
  initialize (CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps) generator =
    let generator' = sGeneratorToDevice device generator
        multiHeadAttention = IxStateT . initialize @multiHeadAttention $ MultiHeadAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNorm = IxStateT . initialize @layerNorm $ case style of
          ST5 -> layerNormWithoutBiasSpec
          SByT5 -> layerNormWithoutBiasSpec
          SBART -> layerNormWithBiasSpec
          SMBART -> layerNormWithBiasSpec
          SPegasus -> layerNormWithBiasSpec
          SBERT -> undefined
          SRoBERTa -> undefined
          SGPT2 -> undefined
        dropout = IxStateT . initialize $ Dropout dropoutP
     in runIxStateT
          ( (GCrossAttention <<$>> multiHeadAttention <<*>> layerNorm <<*>> dropout)
              >>>= (ireturn . CrossAttention)
          )
          generator'

instance
  SingI style =>
  HasStateDict
    (CrossAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
  where
  fromStateDict (CrossAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps) k =
    let multiHeadAttentionSpec = MultiHeadAttentionSpec style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP
        multiHeadAttention ST5 = fromStateDict multiHeadAttentionSpec (k <> "EncDecAttention.")
        multiHeadAttention SByT5 = fromStateDict multiHeadAttentionSpec (k <> "EncDecAttention.")
        multiHeadAttention SBART = fromStateDict multiHeadAttentionSpec (k <> "encoder_attn.")
        multiHeadAttention SMBART = fromStateDict multiHeadAttentionSpec (k <> "encoder_attn.")
        multiHeadAttention SPegasus = fromStateDict multiHeadAttentionSpec (k <> "encoder_attn.")
        multiHeadAttention SBERT = undefined
        multiHeadAttention SRoBERTa = undefined
        multiHeadAttention SGPT2 = undefined
        layerNormWithoutBiasSpec = LayerNormSpec SWithoutBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNormWithBiasSpec = LayerNormSpec SWithBias gradient device dataType (SShape $ queryEmbedDim :|: SNil) eps
        layerNorm ST5 = fromStateDict layerNormWithoutBiasSpec (k <> "layer_norm.")
        layerNorm SByT5 = fromStateDict layerNormWithoutBiasSpec (k <> "layer_norm.")
        layerNorm SBART = fromStateDict layerNormWithBiasSpec (k <> "encoder_attn_layer_norm.")
        layerNorm SMBART = fromStateDict layerNormWithBiasSpec (k <> "encoder_attn_layer_norm.")
        layerNorm SPegasus = fromStateDict layerNormWithBiasSpec (k <> "encoder_attn_layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = fromStateDict (Dropout dropoutP) k
     in CrossAttention
          <$> ( GCrossAttention
                  <$> multiHeadAttention (sing @style)
                  <*> layerNorm (sing @style)
                  <*> dropout (sing @style)
              )
  toStateDict k (CrossAttention GCrossAttention {..}) =
    let multiHeadAttention ST5 = toStateDict (k <> "EncDecAttention.")
        multiHeadAttention SByT5 = toStateDict (k <> "EncDecAttention.")
        multiHeadAttention SBART = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SMBART = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SPegasus = toStateDict (k <> "encoder_attn.")
        multiHeadAttention SBERT = undefined
        multiHeadAttention SRoBERTa = undefined
        multiHeadAttention SGPT2 = undefined
        layerNorm ST5 = toStateDict (k <> "layer_norm.")
        layerNorm SByT5 = toStateDict (k <> "layer_norm.")
        layerNorm SBART = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SMBART = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SPegasus = toStateDict (k <> "encoder_attn_layer_norm.")
        layerNorm SBERT = undefined
        layerNorm SRoBERTa = undefined
        layerNorm SGPT2 = undefined
        dropout _ = toStateDict k
     in do
          () <- multiHeadAttention (sing @style) caMultiHeadAttention
          () <- layerNorm (sing @style) caLayerNorm
          () <- dropout (sing @style) caDropout
          pure ()

-- | 'HasForward' instance for @CrossAttention 'T5@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    HasForward
      (CALayerNormF 'T5 gradient device dataType queryEmbedDim)
      query
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (CAMultiheadAttentionF 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
      (layerNormOutput, key, key, attentionBias)
      generatorDevice
      mhaOutput
      device2,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutputDevice ~ (device1 <+> device2)
  ) =>
  HasForward
    (CrossAttention 'T5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    (query, key, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)

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
      dropoutP = 0.0
      eps = 1e-6
  let g = sMkGenerator device 0
  (ca, g') <-
    initialize
      (CrossAttentionSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps)
      g
  ca' <- flip evalStateT Map.empty $ do
    toStateDict "ca." ca
    fromStateDict
      (CrossAttentionSpec ST5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps)
      "ca."
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @4
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward ca' (query, key, attentionBias) g'
  pure output

-- | 'HasForward' instance for @CrossAttention 'ByT5@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    HasForward
      (CALayerNormF 'ByT5 gradient device dataType queryEmbedDim)
      query
      generatorDevice
      layerNormOutput
      generatorDevice,
    HasForward
      (CAMultiheadAttentionF 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
      (layerNormOutput, key, key, attentionBias)
      generatorDevice
      mhaOutput
      device2,
    query ~ Tensor gradient0 layout0 device0 dataType0 shape0,
    mhaOutput ~ Tensor gradient1 layout1 device1 dataType1 shape1,
    output
      ~ Tensor
          (gradient0 <|> gradient1)
          (layout0 <+> layout1)
          (device0 <+> device1 <+> device2)
          (dataType0 <+> dataType1)
          (BroadcastShapesF shape0 shape1),
    generatorOutputDevice ~ (device1 <+> device2)
  ) =>
  HasForward
    (CrossAttention 'ByT5 gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    (query, key, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @CrossAttenton 'BART@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--   caLayerNorm
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    key ~ Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim)
      (query, key, key, attentionBias)
      generatorDevice
      ( Tensor
          (gradient <|> queryGradient <|> keyGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice),
    output
      ~ Tensor
          (gradient <|> queryGradient <|> keyGradient <|> attentionBiasGradient)
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              (BroadcastShapesF queryShape mhaOutputShape)
          ),
    generatorOutputDevice ~ (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (CrossAttention 'BART gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    (query, key, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)
        >>>= IxStateT . forward (caLayerNorm ca)

-- | 'HasForward' instance for @CrossAttention 'Pegasus@.
--
-- @
--    ┌───────┐  ┌─────┐  ┌───────────────┐
--    │ query │  │ key │  │ attentionBias │
--    └───┬───┘  └──┬──┘  └───────┬───────┘
--        │         │             │
-- ┌──────┤         │             │
-- │      │         │             │
-- │      ▼         │             │
-- │ caLayerNorm    │             │
-- │      │         │             │
-- │      │      ┌──┴──┐          │
-- │      │      │     │          │
-- │      ▼      ▼     ▼          │
-- │   caMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │         caDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @
instance
  ( SGetDim queryEmbedDim,
    query ~ Tensor queryGradient queryLayout queryDevice queryDataType queryShape,
    key ~ Tensor keyGradient keyLayout keyDevice keyDataType keyShape,
    attentionBias ~ Tensor attentionBiasGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    normedQueryGradient ~ (gradient <|> queryGradient),
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithBiasF ('Shape '[queryEmbedDim]) ('Shape '[queryEmbedDim]) queryShape,
    normedQuery ~ Tensor normedQueryGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
    HasForward
      (MultiHeadAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim)
      ( normedQuery,
        key,
        key,
        attentionBias
      )
      generatorDevice
      ( Tensor
          (queryGradient <|> gradient <|> keyGradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice),
    output
      ~ Tensor
          (queryGradient <|> gradient <|> keyGradient <|> attentionBiasGradient)
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutputDevice ~ (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (CrossAttention 'Pegasus gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim)
    (query, key, attentionBias)
    generatorDevice
    output
    generatorOutputDevice
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxStateT $
      ireturn query
        >>>= IxStateT . forward (caLayerNorm ca)
        >>>= (\query' -> IxStateT $ forward (caMultiHeadAttention ca) (query', key, key, attentionBias))
        >>>= IxStateT . forward (caDropout ca)
        >>>= ireturn . (query `add`)
