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

module Torch.GraduallyTyped.NN.Transformer.CrossAttention where

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
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (HasInitializeMultiHeadAttentionC, MultiHeadAttention, lookupMultiHeadAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim, KnownShape, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor (TransposeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
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
      caMultiheadAttention :: mha,
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
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  CrossAttention ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP.
    GCrossAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP ->
    CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP

type GCrossAttentionF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GCrossAttention
    (CAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    (CALayerNormF style device dataType queryEmbedDim)
    (CADropoutF style dropoutP)

type family
  CAMultiheadAttentionF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  CAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP =
    MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP

type family
  CALayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  CALayerNormF 'T5 device dataType queryEmbedDim =
    LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])

type family
  CADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  CADropoutF _ dropoutP =
    Dropout dropoutP

type HasInitializeCrossAttentionC
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))),
    WithDimC keyEmbedDim (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))
  )

instance
  ( HasInitializeCrossAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
    Scalar dropoutP,
    multiHeadAttention ~ CAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP,
    HasInitialize multiHeadAttention,
    InitializeF multiHeadAttention ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF keyEmbedDim (dropoutP -> Generator device -> (multiHeadAttention, Generator device))))))))),
    HasInitializeMultiHeadAttentionC multiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP,
    layerNorm ~ CALayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDeviceC device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDataTypeC dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm)),
    WithDimsC '[queryEmbedDim] (Double -> layerNorm),
    dropout ~ CADropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout)
  ) =>
  HasInitialize (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
  where
  type
    InitializeF (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( WithDimF
                        embedDim
                        ( WithDimF
                            queryEmbedDim
                            ( WithDimF
                                keyEmbedDim
                                (dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device))
                            )
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
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @queryEmbedDim $
                          \queryEmbedDim ->
                            withDim @keyEmbedDim @(dropoutP -> Double -> Generator device -> (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP, Generator device)) $
                              \keyEmbedDim ->
                                go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP eps = runState $ do
        multiHeadAttention <-
          state $
            withoutDim @keyEmbedDim @(dropoutP -> Generator device -> (multiHeadAttention, Generator device))
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @multiHeadAttention
                                          )
                                          deviceType
                                      )
                                      dType
                                  )
                                  headDim
                              )
                              headEmbedDim
                          )
                          embedDim
                      )
                      queryEmbedDim
                  )
                  keyEmbedDim
              )
              keyEmbedDim
              dropoutP
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
        pure . CrossAttention $ GCrossAttention multiHeadAttention layerNorm dropout

lookupCrossAttention ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim keyEmbedDim,
    Scalar dropoutP
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (CrossAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
lookupCrossAttention dropoutP eps prefix =
  let crossAttention _ = lookupMultiHeadAttention dropoutP (prefix <> "EncDecAttention.")
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
   in CrossAttention
        <$> ( GCrossAttention
                <$> crossAttention (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
            )

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
  ( KnownDim queryEmbedDim,
    KnownShape queryShape,
    Scalar dropoutP,
    normedQueryLayout ~ ('Layout 'Dense <+> queryLayout),
    normedQueryDevice ~ (device <+> queryDevice),
    normedQueryDataType ~ (dataType <+> queryDataType),
    normedQueryShape ~ LayerNormWithoutBiasF ('Shape '[queryEmbedDim]) queryShape,
    HasForward
      (MultiHeadAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim keyEmbedDim dropoutP)
      ( Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> keyLayout <+> attentionBiasLayout)
          (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> keyDataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (queryDevice <+> device <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (CrossAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (CrossAttention ca) (query, key, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward (caLayerNorm ca)
        >>>= (\query' -> IxState $ forward (caMultiheadAttention ca) (query', key, key, attentionBias))
        >>>= IxState . forward (caDropout ca)
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
-- │  bcaMultiheadAttention◄──────┘
-- │             │
-- │             ▼
-- │        bcaDropout
-- │             │
-- └────►add◄────┘
--        │
--        ▼
--  bcaLayerNorm
--        │
--        ▼
--    ┌───────┐
--    │ query │
--    └───────┘
-- @