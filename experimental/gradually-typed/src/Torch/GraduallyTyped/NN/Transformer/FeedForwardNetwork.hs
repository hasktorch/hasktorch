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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (sing))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), GeluNew (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, Name (..), SDim, SShape (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, SGetShape, Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Generic transformer feed-forward network.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerFeedForwardNetwork'.
data
  GTransformerFeedForwardNetwork
    (inputWeight1 :: Type)
    (inputWeight2 :: Type)
    (outputWeight :: Type)
    (activation :: Type)
    (activationDropout :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GTransformerFeedForwardNetwork ::
    forall inputWeight1 inputWeight2 outputWeight activation activationDropout layerNorm dropout.
    { -- | first input weight
      ffnInputWeight1 :: inputWeight1,
      -- | second input weight
      ffnInputWeight2 :: inputWeight2,
      -- | output weight
      ffnOutputWeight :: outputWeight,
      -- | activation
      ffnActivation :: activation,
      -- | activation dropout
      ffnActivationDropout :: activationDropout,
      -- | feed-forward layer norm
      ffnLayoutNorm :: layerNorm,
      -- | feed-forward dropout
      ffnDropout :: dropout
    } ->
    GTransformerFeedForwardNetwork inputWeight1 inputWeight2 outputWeight activation activationDropout layerNorm dropout

-- | Transformer feed-forward network.
data
  TransformerFeedForwardNetwork
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerFeedForwardNetwork ::
    forall style device dataType queryEmbedDim ffnDim dropoutP.
    GTransformerFeedForwardNetworkF style device dataType queryEmbedDim ffnDim dropoutP ->
    TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP

type GTransformerFeedForwardNetworkF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GTransformerFeedForwardNetwork
    (FFNInputWeight1F style device dataType queryEmbedDim ffnDim)
    (FFNInputWeight2F style device dataType queryEmbedDim ffnDim)
    (FFNOutputWeightF style device dataType queryEmbedDim ffnDim)
    (FFNActivationF style)
    (FFNActivationDropoutF style dropoutP)
    (FFNLayerNormF style device dataType queryEmbedDim)
    (FFNDropoutF style dropoutP)

type family
  FFNInputWeight1F
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputWeight1F 'T5 device dataType queryEmbedDim ffnDim = Linear 'WithoutBias device dataType queryEmbedDim ffnDim
  FFNInputWeight1F 'ByT5 device dataType queryEmbedDim ffnDim = FFNInputWeight1F 'T5 device dataType queryEmbedDim ffnDim
  FFNInputWeight1F _ device dataType queryEmbedDim ffnDim = Linear 'WithBias device dataType queryEmbedDim ffnDim

type family
  FFNInputWeight2F
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputWeight2F 'ByT5 device dataType queryEmbedDim ffnDim = Linear 'WithoutBias device dataType queryEmbedDim ffnDim
  FFNInputWeight2F _ device dataType queryEmbedDim ffnDim = ()

type family
  FFNOutputWeightF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNOutputWeightF 'T5 device dataType queryEmbedDim ffnDim = Linear 'WithoutBias device dataType ffnDim queryEmbedDim
  FFNOutputWeightF 'ByT5 device dataType queryEmbedDim ffnDim = FFNOutputWeightF 'T5 device dataType queryEmbedDim ffnDim
  FFNOutputWeightF _ device dataType queryEmbedDim ffnDim = Linear 'WithBias device dataType ffnDim queryEmbedDim

type family
  FFNActivationF
    (style :: TransformerStyle) ::
    Type
  where
  FFNActivationF 'T5 = Relu
  FFNActivationF 'ByT5 = GeluNew
  FFNActivationF 'BART = Gelu
  FFNActivationF 'MBART = Gelu
  FFNActivationF 'Pegasus = Relu
  FFNActivationF 'BERT = Gelu
  FFNActivationF 'RoBERTa = Gelu
  FFNActivationF 'GPT2 = Gelu

type family
  FFNActivationDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNActivationDropoutF 'T5 dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'ByT5 dropoutP = FFNActivationDropoutF 'T5 dropoutP
  FFNActivationDropoutF 'BART dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'MBART dropoutP = FFNActivationDropoutF 'BART dropoutP
  FFNActivationDropoutF 'Pegasus dropoutP = FFNActivationDropoutF 'BART dropoutP
  FFNActivationDropoutF 'BERT _ = ()
  FFNActivationDropoutF 'RoBERTa dropoutP = FFNActivationDropoutF 'BERT dropoutP
  FFNActivationDropoutF 'GPT2 _ = ()

type family
  FFNLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNLayerNormF 'T5 device dataType queryEmbedDim = LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])
  FFNLayerNormF 'ByT5 device dataType queryEmbedDim = FFNLayerNormF 'T5 device dataType queryEmbedDim
  FFNLayerNormF _ device dataType queryEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])

type family
  FFNDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNDropoutF _ dropoutP = Dropout dropoutP

type family
  HasInitializeFFNInputWeight2InputF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  HasInitializeFFNInputWeight2InputF 'T5 _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'ByT5 device dataType queryEmbedDim ffnDim = (SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim)
  HasInitializeFFNInputWeight2InputF 'BART _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'MBART device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BART device dataType queryEmbedDim ffnDim
  HasInitializeFFNInputWeight2InputF 'Pegasus device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BART device dataType queryEmbedDim ffnDim
  HasInitializeFFNInputWeight2InputF 'BERT _ _ _ _ = ()
  HasInitializeFFNInputWeight2InputF 'RoBERTa device dataType queryEmbedDim ffnDim = HasInitializeFFNInputWeight2InputF 'BERT device dataType queryEmbedDim ffnDim

type family
  HasInitializeFFNActivationDropoutInputF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  HasInitializeFFNActivationDropoutInputF 'T5 dropoutP = dropoutP
  HasInitializeFFNActivationDropoutInputF 'ByT5 dropoutP = HasInitializeFFNActivationDropoutInputF 'T5 dropoutP
  HasInitializeFFNActivationDropoutInputF 'BART dropoutP = dropoutP
  HasInitializeFFNActivationDropoutInputF 'MBART dropoutP = HasInitializeFFNActivationDropoutInputF 'BART dropoutP
  HasInitializeFFNActivationDropoutInputF 'Pegasus dropoutP = HasInitializeFFNActivationDropoutInputF 'BART dropoutP
  HasInitializeFFNActivationDropoutInputF 'BERT _ = ()
  HasInitializeFFNActivationDropoutInputF 'RoBERTa dropoutP = HasInitializeFFNActivationDropoutInputF 'BERT dropoutP

instance
  ( SingI style,
    inputWeight1 ~ FFNInputWeight1F style device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight1 (SDevice device, SDataType dataType, SDim queryEmbedDim, SDim ffnDim) generator generator',
    inputWeight2 ~ FFNInputWeight2F style device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight2 (HasInitializeFFNInputWeight2InputF style device dataType queryEmbedDim ffnDim) generator' generator'',
    outputWeight ~ FFNOutputWeightF style device dataType queryEmbedDim ffnDim,
    HasInitialize outputWeight (SDevice device, SDataType dataType, SDim ffnDim, SDim queryEmbedDim) generator'' generator''',
    activation ~ FFNActivationF style,
    HasInitialize activation () generator''' generator''',
    activationDropout ~ FFNActivationDropoutF style dropoutP,
    HasInitialize activationDropout (HasInitializeFFNActivationDropoutInputF style dropoutP) generator''' generator''',
    layerNorm ~ FFNLayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm (SDevice device, SDataType dataType, SShape ('Shape '[queryEmbedDim]), Double) generator''' generator''',
    dropout ~ FFNDropoutF style dropoutP,
    HasInitialize dropout dropoutP generator''' generator'''
  ) =>
  HasInitialize
    (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
    ( SDevice device,
      SDataType dataType,
      SDim queryEmbedDim,
      SDim ffnDim,
      dropoutP,
      Double
    )
    generator
    generator'''
  where
  initialize (device, dataType, queryEmbedDim, ffnDim, dropoutP, eps) =
    let inputWeight1 = IxState . initialize $ (device, dataType, queryEmbedDim, ffnDim)
        inputWeight2 = IxState . initialize $
          case sing @style of
            ST5 -> ()
            SByT5 -> (device, dataType, queryEmbedDim, ffnDim)
            SBART -> ()
            SMBART -> ()
            SPegasus -> ()
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        outputWeight = IxState . initialize $ (device, dataType, ffnDim, queryEmbedDim)
        activation = IxState . initialize $ ()
        activationDropout = IxState . initialize $
          case sing @style of
            ST5 -> dropoutP
            SByT5 -> dropoutP
            SBART -> dropoutP
            SMBART -> dropoutP
            SPegasus -> dropoutP
            SBERT -> ()
            SRoBERTa -> ()
            SGPT2 -> undefined
        layerNorm = IxState . initialize $ (device, dataType, SShape $ queryEmbedDim :|: SNil, eps)
        dropout = IxState . initialize $ dropoutP
     in runIxState $
          ( GTransformerFeedForwardNetwork
              <<$>> inputWeight1
              <<*>> inputWeight2
              <<*>> outputWeight
              <<*>> activation
              <<*>> activationDropout
              <<*>> layerNorm
              <<*>> dropout
          )
            >>>= ireturn . TransformerFeedForwardNetwork

lookupTransformerFeedForwardNetwork ::
  forall style device dataType queryEmbedDim ffnDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim queryEmbedDim,
    KnownDim ffnDim,
    Scalar dropoutP
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
lookupTransformerFeedForwardNetwork dropoutP eps prefix =
  let inputWeight1 ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wi.weight")
      inputWeight1 SByT5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wi_0.weight")
      inputWeight1 SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc1.weight")
          <*> lookupTensor (prefix <> "fc1.bias")
      inputWeight1 SMBART = undefined
      inputWeight1 SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc1.weight")
          <*> lookupTensor (prefix <> "fc1.bias")
      inputWeight1 SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "intermediate.dense.weight")
          <*> lookupTensor (prefix <> "intermediate.dense.bias")
      inputWeight1 SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "intermediate.dense.weight")
          <*> lookupTensor (prefix <> "intermediate.dense.bias")
      inputWeight1 SGPT2 = undefined
      inputWeight2 ST5 = pure ()
      inputWeight2 SByT5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wi_1.weight")
      inputWeight2 SBART = pure ()
      inputWeight2 SMBART = pure ()
      inputWeight2 SPegasus = pure ()
      inputWeight2 SBERT = pure ()
      inputWeight2 SRoBERTa = pure ()
      inputWeight2 SGPT2 = pure ()
      outputWeight ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wo.weight")
      outputWeight SByT5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wo.weight")
      outputWeight SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc2.weight")
          <*> lookupTensor (prefix <> "fc2.bias")
      outputWeight SMBART = undefined
      outputWeight SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc2.weight")
          <*> lookupTensor (prefix <> "fc2.bias")
      outputWeight SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outputWeight SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outputWeight SGPT2 = undefined
      activation ST5 = pure @m Relu
      activation SByT5 = pure @m GeluNew
      activation SBART = pure @m Gelu
      activation SMBART = undefined
      activation SPegasus = pure @m Relu
      activation SBERT = pure @m Gelu
      activation SRoBERTa = pure @m Gelu
      activation SGPT2 = undefined
      activationDropout ST5 = pure (Dropout dropoutP)
      activationDropout SByT5 = pure (Dropout dropoutP)
      activationDropout SBART = pure (Dropout dropoutP)
      activationDropout SMBART = undefined
      activationDropout SPegasus = pure (Dropout dropoutP)
      activationDropout SBERT = pure ()
      activationDropout SRoBERTa = pure ()
      activationDropout SGPT2 = undefined
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      layerNorm SByT5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      layerNorm SBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> lookupTensor (prefix <> "final_layer_norm.bias")
          <*> pure eps
      layerNorm SMBART = undefined
      layerNorm SPegasus =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> lookupTensor (prefix <> "final_layer_norm.bias")
          <*> pure eps
      layerNorm SBERT =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "output.LayerNorm.weight")
          <*> lookupTensor (prefix <> "output.LayerNorm.bias")
          <*> pure eps
      layerNorm SRoBERTa =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "output.LayerNorm.weight")
          <*> lookupTensor (prefix <> "output.LayerNorm.bias")
          <*> pure eps
      layerNorm SGPT2 = undefined
      dropout _ = pure (Dropout dropoutP)
   in TransformerFeedForwardNetwork
        <$> ( GTransformerFeedForwardNetwork
                <$> inputWeight1 (sing @style)
                <*> inputWeight2 (sing @style)
                <*> outputWeight (sing @style)
                <*> activation (sing @style)
                <*> activationDropout (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
            )

type family
  FeedForwardNetworkOutputShape
    (style :: TransformerStyle)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryShape :: Shape [Dim (Name Symbol) (Size Nat)]) ::
    Shape [Dim (Name Symbol) (Size Nat)]
  where
  FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape =
    BroadcastShapesF
      queryShape
      ( LinearWithoutBiasF
          ('Shape '[queryEmbedDim, ffnDim])
          ( LinearWithoutBiasF
              ('Shape '[ffnDim, queryEmbedDim])
              ( LayerNormWithoutBiasF
                  ('Shape '[queryEmbedDim])
                  queryShape
              )
          )
      )
  FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape = FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape
  FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape =
    BroadcastShapesF
      queryShape
      ( LinearWithBiasF
          ('Shape '[queryEmbedDim, ffnDim])
          ('Shape '[queryEmbedDim])
          ( LinearWithBiasF
              ('Shape '[ffnDim, queryEmbedDim])
              ('Shape '[ffnDim])
              ( LayerNormWithBiasF
                  ('Shape '[queryEmbedDim])
                  ('Shape '[queryEmbedDim])
                  queryShape
              )
          )
      )
  FeedForwardNetworkOutputShape _ queryEmbedDim ffnDim queryShape =
    LayerNormWithBiasF
      ('Shape '[queryEmbedDim])
      ('Shape '[queryEmbedDim])
      ( BroadcastShapesF
          queryShape
          ( LinearWithBiasF
              ('Shape '[queryEmbedDim, ffnDim])
              ('Shape '[queryEmbedDim])
              ( LinearWithBiasF
                  ('Shape '[ffnDim, queryEmbedDim])
                  ('Shape '[ffnDim])
                  queryShape
              )
          )
      )

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'T5@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'T5 queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'T5 device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnLayoutNorm
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'ByT5@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'ByT5 queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'ByT5 device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    let activate query' =
          ireturn query'
            >>>= IxState . forward ffnInputWeight1
            >>>= IxState . forward ffnActivation
        gate query' = (*) <<$>> activate query' <<*>> (IxState . forward ffnInputWeight2 $ query')
     in runIxState $
          ireturn query
            >>>= IxState . forward ffnLayoutNorm
            >>>= gate
            >>>= IxState . forward ffnActivation
            >>>= IxState . forward ffnActivationDropout
            >>>= IxState . forward ffnOutputWeight
            >>>= IxState . forward ffnDropout
            >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'BART@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'BART queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> ((device <+> queryDevice) <+> generatorDevice)) <+> ((device <+> queryDevice) <+> generatorDevice))
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BART device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayoutNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'BERT@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'BERT queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'BERT device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayoutNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'RoBERTa@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--      ffnLayerNorm
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout)
          (device <+> queryDevice <+> generatorDevice)
          (dataType <+> queryDataType)
          (FeedForwardNetworkOutputShape 'RoBERTa queryEmbedDim ffnDim queryShape),
    generatorOutput
      ~ Generator ((device <+> (device <+> queryDevice)) <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'RoBERTa device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
        >>>= IxState . forward ffnLayoutNorm

-- | 'HasForward' instance for @TransformerFeedForwardNetwork 'Pegasus@.
--
-- @
--       ┌───────┐
--       │ query ├───────┐
--       └───┬───┘       │
--           │           │
--           ▼           │
--      ffnLayerNorm     │
--           ▼           │
--     ffnInputWeight    │
--           ▼           │
--     ffnActivation     │
--           ▼           │
--  ffnActivationDropout │
--           ▼           │
--    ffnOutputWeight    │
--           ▼           │
--       ffnDropout      │
--           │           │
--           ▼           │
--          add◄─────────┘
--           │
--           ▼
--       ┌───────┐
--       │ query │
--       └───────┘
-- @
instance
  ( SGetShape queryShape,
    SGetDim queryEmbedDim,
    Scalar dropoutP,
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense)
          (queryDevice <+> device <+> generatorDevice)
          (queryDataType <+> dataType)
          (FeedForwardNetworkOutputShape 'Pegasus queryEmbedDim ffnDim queryShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> generatorDevice)
  ) =>
  HasForward
    (TransformerFeedForwardNetwork 'Pegasus device dataType queryEmbedDim ffnDim dropoutP)
    (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (TransformerFeedForwardNetwork GTransformerFeedForwardNetwork {..}) query =
    runIxState $
      ireturn query
        >>>= IxState . forward ffnLayoutNorm
        >>>= IxState . forward ffnInputWeight1
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
