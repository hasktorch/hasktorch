{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
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

module Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Constraint, Type)
import Data.Singletons (SingI, sing)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Activation (Gelu (..), Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearWithBiasC, HasInitializeLinearWithoutBiasC, Linear (..))
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithBiasC, HasInitializeLayerNormWithoutBiasC, LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim, KnownShape, Name (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
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

type HasInitializeTransformerFeedForwardNetworkC style device dataType queryEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDataTypeC dataType (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device))
  )

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
  HasInitializeFFNActivationDropoutF activationDropout 'BERT dropoutP =
    ()
  HasInitializeFFNActivationDropoutF activationDropout 'Pegasus dropoutP =
    ( HasInitialize activationDropout,
      InitializeF activationDropout ~ (dropoutP -> activationDropout)
    )

instance
  ( SingI style,
    HasInitializeTransformerFeedForwardNetworkC style device dataType queryEmbedDim ffnDim dropoutP,
    inputWeight ~ FFNInputWeightF style device dataType queryEmbedDim ffnDim,
    HasInitialize inputWeight,
    InitializeF inputWeight ~ WithDeviceF device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (Generator device -> (inputWeight, Generator device))))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF ffnDim (Generator device -> (inputWeight, Generator device))))),
    WithDataTypeC dataType (WithDimF queryEmbedDim (WithDimF ffnDim (Generator device -> (inputWeight, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (Generator device -> (inputWeight, Generator device))),
    WithDimC ffnDim (Generator device -> (inputWeight, Generator device)),
    outputWeight ~ FFNOutputWeightF style device dataType queryEmbedDim ffnDim,
    HasInitialize outputWeight,
    InitializeF outputWeight ~ WithDeviceF device (WithDataTypeF dataType (WithDimF ffnDim (WithDimF queryEmbedDim (Generator device -> (outputWeight, Generator device))))),
    WithDeviceC device (WithDataTypeF dataType (WithDimF ffnDim (WithDimF queryEmbedDim (Generator device -> (outputWeight, Generator device))))),
    WithDataTypeC dataType (WithDimF ffnDim (WithDimF queryEmbedDim (Generator device -> (outputWeight, Generator device)))),
    WithDimC ffnDim (WithDimF queryEmbedDim (Generator device -> (outputWeight, Generator device))),
    WithDimC ffnDim (WithDimF queryEmbedDim (Generator device -> (outputWeight, Generator device))),
    WithDimC queryEmbedDim (Generator device -> (outputWeight, Generator device)),
    activation ~ FFNActivationF style,
    HasInitialize activation,
    InitializeF activation ~ activation,
    activationDropout ~ FFNActivationDropoutF style dropoutP,
    HasInitializeFFNActivationDropoutF activationDropout style dropoutP,
    layerNorm ~ FFNLayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDeviceC device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDataTypeC dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm)),
    WithDimsC '[queryEmbedDim] (Double -> layerNorm),
    dropout ~ FFNDropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout)
  ) =>
  HasInitialize (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                queryEmbedDim
                ( WithDimF
                    ffnDim
                    ( dropoutP ->
                      Double ->
                      Generator device ->
                      ( TransformerFeedForwardNetwork
                          style
                          device
                          dataType
                          queryEmbedDim
                          ffnDim
                          dropoutP,
                        Generator device
                      )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @queryEmbedDim $
              \queryEmbedDim ->
                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerFeedForwardNetwork style device dataType queryEmbedDim ffnDim dropoutP, Generator device)) $
                  \ffnDim ->
                    go deviceType dType queryEmbedDim ffnDim
    where
      go deviceType dType queryEmbedDim ffnDim dropoutP eps = runState $ do
        inputWeight <-
          state $
            withoutDim @ffnDim @(Generator device -> (inputWeight, Generator device))
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @inputWeight
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              ffnDim
        outputWeight <-
          state $
            withoutDim @queryEmbedDim @(Generator device -> (outputWeight, Generator device))
              ( withoutDim @ffnDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @outputWeight
                          )
                          deviceType
                      )
                      dType
                  )
                  ffnDim
              )
              queryEmbedDim
        let activation = initialize @activation
        let activationDropout = case sing @style of
              ST5 -> initialize @activationDropout dropoutP
              SBART -> initialize @activationDropout dropoutP
              SBERT -> ()
              SPegasus -> initialize @activationDropout dropoutP
        let layerNorm =
              withoutShape @('Shape '[queryEmbedDim]) @(Double -> layerNorm)
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @layerNorm
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
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
      activation SBART = pure @m Gelu
      activation SPegasus = pure @m Relu
      activationDropout ST5 = pure (initialize @(Dropout dropoutP) dropoutP)
      activationDropout SBERT = pure ()
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
  ( KnownShape queryShape,
    KnownDim queryEmbedDim,
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
  ( KnownShape queryShape,
    KnownDim queryEmbedDim,
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
  ( KnownShape queryShape,
    KnownDim queryEmbedDim,
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
  ( KnownShape queryShape,
    KnownDim queryEmbedDim,
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
