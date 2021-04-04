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
{-# LANGUAGE TupleSections #-}
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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL9
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL9C #-}

module Torch.GraduallyTyped.NN.Transformer.SelfAttention where

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
import Torch.GraduallyTyped.NN.Functional.Linear (LinearWithBiasF, LinearWithoutBiasF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormWithBiasF, LayerNormWithoutBiasF)
import Torch.GraduallyTyped.NN.Linear (Linear)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (HasInitializeMultiHeadAttentionC, MultiHeadAttention, lookupMultiHeadAttention)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..), WithDimsC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor (ReshapeF)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (Unify, type (<+>), type (<|>))

-- | Generic self-attention layer.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'SelfAttention'.
data
  GSelfAttention
    (mha :: Type)
    (layerNorm :: Type)
    (dropout :: Type)
    (dense :: Type)
  where
  GSelfAttention ::
    forall mha layerNorm dropout dense.
    { -- | self-attention
      saMultiheadAttention :: mha,
      -- | layer norm
      saLayerNorm :: layerNorm,
      -- | dropout
      saDropout :: dropout,
      -- | dense
      saDense :: dense
    } ->
    GSelfAttention mha layerNorm dropout dense

-- | Self-attention layer.
newtype
  SelfAttention
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SelfAttention ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP.
    GSelfAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP ->
    SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP

type GSelfAttentionF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GSelfAttention
    (SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (SALayerNormF style device dataType queryEmbedDim)
    (SADropoutF style dropoutP)
    (SADenseF style device dataType queryEmbedDim)

type family
  SAMultiheadAttentionF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
    MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP

type family
  SALayerNormF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SALayerNormF 'T5 device dataType queryEmbedDim =
    LayerNorm 'WithoutBias device dataType ('Shape '[queryEmbedDim])
  SALayerNormF 'BERT device dataType queryEmbedDim =
    LayerNorm 'WithBias device dataType ('Shape '[queryEmbedDim])

type family
  SADropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  SADropoutF _ dropoutP = Dropout dropoutP

type family
  SADenseF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SADenseF 'BERT device dataType queryEmbedDim =
    Linear 'WithBias device dataType queryEmbedDim queryEmbedDim
  SADenseF _ _ _ _ =
    ()

type HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)))),
    WithDimC embedDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))),
    WithDimC queryEmbedDim (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
  )

type family
  HasInitializeSADenseF
    (dense :: Type)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Constraint
  where
  HasInitializeSADenseF dense 'BERT device dataType queryEmbedDim =
    ( HasInitialize dense,
      InitializeF dense ~ WithDeviceF device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF queryEmbedDim (Generator device -> (dense, Generator device))))),
      WithDeviceC device (WithDataTypeF dataType (WithDimF queryEmbedDim (WithDimF queryEmbedDim (Generator device -> (dense, Generator device))))),
      WithDataTypeC dataType (WithDimF queryEmbedDim (WithDimF queryEmbedDim (Generator device -> (dense, Generator device)))),
      WithDimC queryEmbedDim (WithDimF queryEmbedDim (Generator device -> (dense, Generator device))),
      WithDimC queryEmbedDim (Generator device -> (dense, Generator device))
    )
  HasInitializeSADenseF dense _ device dataType queryEmbedDim =
    ()

instance
  ( SingI style,
    HasInitializeSelfAttentionC style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    Scalar dropoutP,
    multiHeadAttention ~ SAMultiheadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP,
    HasInitialize multiHeadAttention,
    InitializeF multiHeadAttention ~ WithDeviceF device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF queryEmbedDim (WithDimF queryEmbedDim (dropoutP -> Generator device -> (multiHeadAttention, Generator device))))))))),
    HasInitializeMultiHeadAttentionC multiHeadAttention device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    layerNorm ~ SALayerNormF style device dataType queryEmbedDim,
    HasInitialize layerNorm,
    InitializeF layerNorm ~ WithDeviceF device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDeviceC device (WithDataTypeF dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm))),
    WithDataTypeC dataType (WithDimsF '[queryEmbedDim] (Double -> layerNorm)),
    WithDimsC '[queryEmbedDim] (Double -> layerNorm),
    dropout ~ SADropoutF style dropoutP,
    HasInitialize dropout,
    InitializeF dropout ~ (dropoutP -> dropout),
    dense ~ SADenseF style device dataType queryEmbedDim,
    HasInitializeSADenseF dense style device dataType queryEmbedDim
  ) =>
  HasInitialize (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP) =
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
                            (dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device))
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
                        withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP, Generator device)) $
                          \queryEmbedDim ->
                            go deviceType dType headDim headEmbedDim embedDim queryEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim dropoutP eps = runState $ do
        multiHeadAttention <-
          state $
            withoutDim @queryEmbedDim @(dropoutP -> Generator device -> (multiHeadAttention, Generator device))
              ( withoutDim @queryEmbedDim
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
                  queryEmbedDim
              )
              queryEmbedDim
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
        dense <-
          state $ case sing @style of
            SBERT ->
              withoutDim @queryEmbedDim @(Generator device -> (dense, Generator device))
                ( withoutDim @queryEmbedDim
                    ( withoutDataType @dataType
                        ( withoutDevice @device
                            ( initialize @dense
                            )
                            deviceType
                        )
                        dType
                    )
                    queryEmbedDim
                )
                queryEmbedDim
            ST5 -> ((),)
            SBART -> ((),)
            SPegasus -> ((),)
        pure . SelfAttention $ GSelfAttention multiHeadAttention layerNorm dropout dense

lookupSelfAttention ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP m.
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
    Scalar dropoutP
  ) =>
  dropoutP ->
  Double ->
  String ->
  m (SelfAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
lookupSelfAttention dropoutP eps prefix =
  let selfAttention _ = lookupMultiHeadAttention dropoutP (prefix <> "SelfAttention.")
      layerNorm ST5 =
        LayerNormWithoutBias
          <$> lookupTensor (prefix <> "layer_norm.weight")
          <*> pure eps
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
      dense ST5 = pure @m ()
   in SelfAttention
        <$> ( GSelfAttention
                <$> selfAttention (sing @style)
                <*> layerNorm (sing @style)
                <*> dropout (sing @style)
                <*> dense (sing @style)
            )

-- | 'HasForward' instance for @SelfAttention 'T5@.
--
-- @
-- ┌───────────────┐     ┌───────┐
-- │ attentionBias │     │ query │
-- └───────┬───────┘     └───┬───┘
--         │                 │
--         │           ┌─────┴─────┐
--         │           │           │
--         │           ▼           │
--         │      saLayerNorm      │
--         │           │           │
--         │      ┌────┼────┐      │
--         │      │    │    │      │
--         │      ▼    ▼    ▼      │
--         └─►saMultiheadAttention │
--                     │           │
--                     ▼           │
--                 saDropout       │
--                     │           │
--                     └───►add◄───┘
--                           │
--                           ▼
--                       ┌───────┐
--                       │ query │
--                       └───────┘
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
      (MultiHeadAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      ( Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor 'WithGradient normedQueryLayout normedQueryDevice normedQueryDataType normedQueryShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionBiasLayout)
          (queryDevice <+> device <+> attentionBiasDevice <+> generatorDevice)
          (queryDataType <+> dataType <+> attentionBiasDataType)
          (BroadcastShapesF queryShape mhaOutputShape),
    generatorOutput ~ Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (SelfAttention 'T5 device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention sa) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= IxState . forward (saLayerNorm sa)
        >>>= (\query' -> IxState $ forward (saMultiheadAttention sa) (query', query', query', attentionBias))
        >>>= IxState . forward (saDropout sa)
        >>>= ireturn . (query `add`)

-- | 'HasForward' instance for @SelfAttention 'BART@.
--
-- @
-- ┌───────────────┐      ┌───────┐
-- │ attentionBias │      │ query │
-- └───────┬───────┘      └───┬───┘
--         │                  │
--         │            ┌─────┴─────┐
--         │            │           │
--         │       ┌────┼────┐      │
--         │       │    │    │      │
--         │       ▼    ▼    ▼      │
--         └─►saMultiheadAttention  │
--                      │           │
--                      ▼           │
--                  saDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                       saLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @

-- | 'HasForward' instance for @SelfAttention 'BERT@.
--
-- @
-- ┌───────────────┐      ┌───────┐
-- │ attentionBias │      │ query │
-- └───────┬───────┘      └───┬───┘
--         │                  │
--         │            ┌─────┴─────┐
--         │            │           │
--         │       ┌────┼────┐      │
--         │       │    │    │      │
--         │       ▼    ▼    ▼      │
--         └─►saMultiheadAttention  │
--                      │           │
--                      ▼           │
--                   saDense        │
--                      ▼           │
--                  saDropout       │
--                      │           │
--                      └───►add◄───┘
--                            │
--                            ▼
--                       saLayerNorm
--                            │
--                            ▼
--                        ┌───────┐
--                        │ query │
--                        └───────┘
-- @
instance
  ( KnownDim queryEmbedDim,
    Scalar dropoutP,
    query ~ Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
    attentionBias ~ Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape,
    HasForward
      (MultiHeadAttention 'BERT device dataType headDim headEmbedDim embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
      (query, query, query, attentionBias)
      (Generator generatorDevice)
      ( Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          mhaOutputShape
      )
      (Generator (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)),
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> attentionBiasLayout)
          (device <+> queryDevice <+> attentionBiasDevice <+> generatorDevice)
          (dataType <+> queryDataType <+> attentionBiasDataType)
          ( LayerNormWithBiasF
              ('Shape '[queryEmbedDim])
              ('Shape '[queryEmbedDim])
              ( BroadcastShapesF
                  queryShape
                  ( LinearWithBiasF
                      ('Shape '[queryEmbedDim, queryEmbedDim])
                      ('Shape '[queryEmbedDim])
                      mhaOutputShape
                  )
              )
          ),
    generatorOutput
      ~ Generator ((device <+> (device <+> (queryDevice <+> (attentionBiasDevice <+> generatorDevice)))) <+> (device <+> (queryDevice <+> (attentionBiasDevice <+> generatorDevice))))
  ) =>
  HasForward
    (SelfAttention 'BERT device dataType headDim headEmbedDim embedDim queryEmbedDim dropoutP)
    (query, attentionBias)
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (SelfAttention sa) (query, attentionBias) =
    runIxState $
      ireturn query
        >>>= (\query' -> IxState $ forward (saMultiheadAttention sa) (query', query', query', attentionBias))
        >>>= IxState . forward (saDense sa)
        >>>= IxState . forward (saDropout sa)
        >>>= ireturn . (query `add`)
        >>>= IxState . forward (saLayerNorm sa)
