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
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Constraint, Type)
import Data.Singletons (SingI, sing)
import Data.Singletons.Prelude.List (SList (SNil))
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
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
import Torch.GraduallyTyped.Tensor.Type (Tensor, SGetDim, SGetShape)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Generic transformer feed-forward network.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'TransformerFeedForwardNetwork'.
data
  GTransformerFeedForwardNetwork
    (inputWeight :: Type)
    (outputWeight :: Type)
    (activation :: Type)
    (activationDropout :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
  where
  GTransformerFeedForwardNetwork ::
    forall inputWeight outputWeight activation activationDropout layerNorm dropout.
    { -- | input weight
      ffnInputWeight :: inputWeight,
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
    GTransformerFeedForwardNetwork inputWeight outputWeight activation activationDropout layerNorm dropout

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
    (FFNInputWeightF style device dataType queryEmbedDim ffnDim)
    (FFNOutputWeightF style device dataType queryEmbedDim ffnDim)
    (FFNActivationF style)
    (FFNActivationDropoutF style dropoutP)
    (FFNLayerNormF style device dataType queryEmbedDim)
    (FFNDropoutF style dropoutP)

type family
  FFNInputWeightF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNInputWeightF 'T5 device dataType queryEmbedDim ffnDim = Linear 'WithoutBias device dataType queryEmbedDim ffnDim
  FFNInputWeightF _ device dataType queryEmbedDim ffnDim = Linear 'WithBias device dataType queryEmbedDim ffnDim

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
  FFNOutputWeightF _ device dataType queryEmbedDim ffnDim = Linear 'WithBias device dataType ffnDim queryEmbedDim

type family
  FFNActivationF
    (style :: TransformerStyle) ::
    Type
  where
  FFNActivationF 'T5 = Relu
  FFNActivationF 'BART = Gelu
  FFNActivationF 'BERT = Gelu
  FFNActivationF 'RoBERTa = Gelu
  FFNActivationF 'Pegasus = Relu

type family
  FFNActivationDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNActivationDropoutF 'T5 dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'BART dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'Pegasus dropoutP = Dropout dropoutP
  FFNActivationDropoutF 'BERT _ = ()
  FFNActivationDropoutF 'RoBERTa _ = ()

type family
  FFNLayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  FFNLayerNormF 'T5 device dataType queryEmbedDim = LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])
  FFNLayerNormF _ device dataType queryEmbedDim = LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])

type family
  FFNDropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  FFNDropoutF 'T5 dropoutP = Dropout dropoutP
  FFNDropoutF _ dropoutP = Dropout dropoutP

type family
  HasInitializeFFNActivationDropoutF
    (activationDropout :: Type)
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Constraint
  where
  HasInitializeFFNActivationDropoutF activationDropout 'T5 dropoutP =
    ( HasInitialize activationDropout,
      InitializeF activationDropout ~ (dropoutP -> activationDropout)
    )
  HasInitializeFFNActivationDropoutF activationDropout 'BART dropoutP =
    ( HasInitialize activationDropout,
      InitializeF activationDropout ~ (dropoutP -> activationDropout)
    )
  HasInitializeFFNActivationDropoutF _ 'BERT _ = ()
  HasInitializeFFNActivationDropoutF _ 'RoBERTa _ = ()
  HasInitializeFFNActivationDropoutF activationDropout 'Pegasus dropoutP =
    ( HasInitialize activationDropout,
      InitializeF activationDropout ~ (dropoutP -> activationDropout)
    )

instance
  ( SingI style,
    inputWeight ~ FFNInputWeightF style device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight,
    InitializeF inputWeight ~ (SDevice device -> SDataType dataType -> SDim queryEmbedDim -> SDim ffnDim -> Generator device -> (inputWeight, Generator device)),
    outputWeight ~ FFNOutputWeightF style device dataType queryEmbedDim ffnDim,
    HasInitialize outputWeight,
    InitializeF outputWeight ~ (SDevice device -> SDataType dataType -> SDim ffnDim -> SDim queryEmbedDim -> Generator device -> (outputWeight, Generator device)),
    activation ~ FFNActivationF style,
    HasInitialize activation,
    InitializeF activation ~ activation,
    activationDropout ~ FFNActivationDropoutF style dropoutP,
    layerNorm ~ FFNLayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ (SDevice device -> SDataType dataType -> SShape ('Shape '[queryEmbedDim]) -> Double -> layerNorm),
    dropout ~ FFNDropoutF style dropoutP,
    HasInitialize dropout
  ) =>
  HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim queryEmbedDim ->
      SDim ffnDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device)
  initialize device dataType queryEmbedDim ffnDim dropoutP eps = runState $ do
    inputWeight <- state $ initialize @inputWeight device dataType queryEmbedDim ffnDim
    outputWeight <- state $ initialize @outputWeight device dataType ffnDim queryEmbedDim
    let activation = initialize @activation
    let activationDropout = case sing @style of
          ST5 -> initialize @activationDropout dropoutP
          SBART -> initialize @activationDropout dropoutP
          SBERT -> ()
          SPegasus -> initialize @activationDropout dropoutP
    let layerNorm = initialize @layerNorm device dataType (SShape $ queryEmbedDim :|: SNil) eps
    let dropout = initialize @dropout dropoutP
    pure . TransformerFeedForwardNetwork $ GTransformerFeedForwardNetwork inputWeight outputWeight activation activationDropout layerNorm dropout

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
  let inputWeight ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wi.weight")
      inputWeight SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "intermediate.dense.weight")
          <*> lookupTensor (prefix <> "intermediate.dense.bias")
      inputWeight SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "intermediate.dense.weight")
          <*> lookupTensor (prefix <> "intermediate.dense.bias")
      inputWeight SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc1.weight")
          <*> lookupTensor (prefix <> "fc1.bias")
      inputWeight SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc1.weight")
          <*> lookupTensor (prefix <> "fc1.bias")
      outputWeight ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "DenseReluDense.wo.weight")
      outputWeight SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outputWeight SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outputWeight SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc2.weight")
          <*> lookupTensor (prefix <> "fc2.bias")
      outputWeight SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "fc2.weight")
          <*> lookupTensor (prefix <> "fc2.bias")
      activation ST5 = pure @m Relu
      activation SBERT = pure @m Gelu
      activation SRoBERTa = pure @m Gelu
      activation SBART = pure @m Gelu
      activation SPegasus = pure @m Relu
      activationDropout ST5 = pure (initialize @(Dropout dropoutP) dropoutP)
      activationDropout SBERT = pure ()
      activationDropout SRoBERTa = pure ()
      activationDropout SBART = pure (initialize @(Dropout dropoutP) dropoutP)
      activationDropout SPegasus = pure (initialize @(Dropout dropoutP) dropoutP)
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
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
      layerNorm SBART =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> lookupTensor (prefix <> "final_layer_norm.bias")
          <*> pure eps
      layerNorm SPegasus =
        LayerNormWithBias
          <$> lookupTensor (prefix <> "final_layer_norm.weight")
          <*> lookupTensor (prefix <> "final_layer_norm.bias")
          <*> pure eps
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
   in TransformerFeedForwardNetwork
        <$> ( GTransformerFeedForwardNetwork
                <$> inputWeight (sing @style)
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
        >>>= IxState . forward ffnInputWeight
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
        >>>= IxState . forward ffnInputWeight
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
        >>>= IxState . forward ffnInputWeight
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
        >>>= IxState . forward ffnInputWeight
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
        >>>= IxState . forward ffnInputWeight
        >>>= IxState . forward ffnActivation
        >>>= IxState . forward ffnActivationDropout
        >>>= IxState . forward ffnOutputWeight
        >>>= IxState . forward ffnDropout
        >>>= ireturn . (query `add`)
