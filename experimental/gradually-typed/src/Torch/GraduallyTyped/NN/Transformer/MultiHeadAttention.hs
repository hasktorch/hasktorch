{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator, mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, sGetDim, sUnifyDim, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (ones)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, sReshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, mulScalar)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (..), Tensor (..), checkedDataType, checkedDevice, checkedLayout, checkedShape)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import qualified Torch.Tensor

-- | Generic multi-headed attention layer.
-- Needs to be specialized to a given transformer type, e.g. 'T5'.
-- See 'MultiHeadAttention'.
data
  GMultiHeadAttention
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (qInProj :: Type)
    (kInProj :: Type)
    (vInProj :: Type)
    (outProj :: Type)
    (dropout :: Type)
  where
  GMultiHeadAttention ::
    forall headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout.
    { -- | head dim
      mhaHeadDim :: SDim headDim,
      -- | head embed dim
      mhaHeadEmbedDim :: SDim headEmbedDim,
      -- | embed dim
      mhaEmbedDim :: SDim embedDim,
      -- | in-projection for query
      mhaQInProj :: qInProj,
      -- | in-projection for key
      mhaKInProj :: kInProj,
      -- | in-projection for value
      mhaVInProj :: vInProj,
      -- | out-projection
      mhaOutProj :: outProj,
      -- | dropout
      mhaDropout :: dropout
    } ->
    GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout

-- | Multi-headed attention layer.
newtype
  MultiHeadAttention
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  MultiHeadAttention ::
    forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
    GMultiHeadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
    MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type GMultiHeadAttentionF
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
  (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GMultiHeadAttention
    headDim
    headEmbedDim
    embedDim
    (QInProjF style device dataType queryEmbedDim embedDim)
    (KInProjF style device dataType keyEmbedDim embedDim)
    (VInProjF style device dataType valueEmbedDim embedDim)
    (OutProjF style device dataType embedDim queryEmbedDim)
    (DropoutF style dropoutP)

type family
  QInProjF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  QInProjF 'T5 device dataType queryEmbedDim embedDim = Linear 'WithoutBias device dataType queryEmbedDim embedDim
  QInProjF _ device dataType queryEmbedDim embedDim = Linear 'WithBias device dataType queryEmbedDim embedDim

type family
  KInProjF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  KInProjF 'T5 device dataType keyEmbedDim embedDim = Linear 'WithoutBias device dataType keyEmbedDim embedDim
  KInProjF _ device dataType keyEmbedDim embedDim = Linear 'WithBias device dataType keyEmbedDim embedDim

type family
  VInProjF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  VInProjF 'T5 device dataType valueEmbedDim embedDim = Linear 'WithoutBias device dataType valueEmbedDim embedDim
  VInProjF _ device dataType valueEmbedDim embedDim = Linear 'WithBias device dataType valueEmbedDim embedDim

type family
  OutProjF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  OutProjF 'T5 device dataType embedDim queryEmbedDim = Linear 'WithoutBias device dataType embedDim queryEmbedDim
  OutProjF _ device dataType embedDim queryEmbedDim = Linear 'WithBias device dataType embedDim queryEmbedDim

type family
  DropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  DropoutF _ dropoutP = Dropout dropoutP

instance
  ( qInProj ~ QInProjF style device dataType queryEmbedDim embedDim,
    HasInitialize qInProj,
    InitializeF qInProj ~ (SDevice device -> SDataType dataType -> SDim queryEmbedDim -> SDim embedDim -> Generator device -> (qInProj, Generator device)),
    kInProj ~ KInProjF style device dataType keyEmbedDim embedDim,
    HasInitialize kInProj,
    InitializeF kInProj ~ (SDevice device -> SDataType dataType -> SDim keyEmbedDim -> SDim embedDim -> Generator device -> (kInProj, Generator device)),
    vInProj ~ VInProjF style device dataType valueEmbedDim embedDim,
    HasInitialize vInProj,
    InitializeF vInProj ~ (SDevice device -> SDataType dataType -> SDim valueEmbedDim -> SDim embedDim -> Generator device -> (vInProj, Generator device)),
    outProj ~ OutProjF style device dataType embedDim queryEmbedDim,
    HasInitialize outProj,
    InitializeF outProj ~ (SDevice device -> SDataType dataType -> SDim embedDim -> SDim queryEmbedDim -> Generator device -> (outProj, Generator device)),
    dropout ~ DropoutF style dropoutP,
    HasInitialize dropout,
    GMultiHeadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ~ GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout
  ) =>
  HasInitialize (MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim queryEmbedDim ->
      SDim keyEmbedDim ->
      SDim valueEmbedDim ->
      dropoutP ->
      Generator device ->
      (MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP = runState $ do
    qInProj <- state $ initialize @qInProj device dataType queryEmbedDim embedDim
    kInProj <- state $ initialize @kInProj device dataType keyEmbedDim embedDim
    vInProj <- state $ initialize @vInProj device dataType valueEmbedDim embedDim
    outProj <- state $ initialize @outProj device dataType embedDim queryEmbedDim
    let dropout = initialize @dropout dropoutP
    pure . MultiHeadAttention $ GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout

lookupMultiHeadAttention ::
  forall style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim embedDim,
    KnownDim queryEmbedDim,
    KnownDim keyEmbedDim,
    KnownDim valueEmbedDim,
    Scalar dropoutP
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  dropoutP ->
  String ->
  m (MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
lookupMultiHeadAttention headDim headEmbedDim embedDim dropoutP prefix =
  let qInProj ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "q.weight")
      qInProj SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.query.weight")
          <*> lookupTensor (prefix <> "self.query.bias")
      qInProj SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.query.weight")
          <*> lookupTensor (prefix <> "self.query.bias")
      qInProj SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "q_proj.weight")
          <*> lookupTensor (prefix <> "q_proj.bias")
      qInProj SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "q_proj.weight")
          <*> lookupTensor (prefix <> "q_proj.bias")
      kInProj ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "k.weight")
      kInProj SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.key.weight")
          <*> lookupTensor (prefix <> "self.key.bias")
      kInProj SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.key.weight")
          <*> lookupTensor (prefix <> "self.key.bias")
      kInProj SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "k_proj.weight")
          <*> lookupTensor (prefix <> "k_proj.bias")
      kInProj SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "k_proj.weight")
          <*> lookupTensor (prefix <> "k_proj.bias")
      vInProj ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "v.weight")
      vInProj SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.value.weight")
          <*> lookupTensor (prefix <> "self.value.bias")
      vInProj SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "self.value.weight")
          <*> lookupTensor (prefix <> "self.value.bias")
      vInProj SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "v_proj.weight")
          <*> lookupTensor (prefix <> "v_proj.bias")
      vInProj SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "v_proj.weight")
          <*> lookupTensor (prefix <> "v_proj.bias")
      outProj ST5 =
        LinearWithoutBias
          <$> lookupTensor (prefix <> "o.weight")
      outProj SBERT =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outProj SRoBERTa =
        LinearWithBias
          <$> lookupTensor (prefix <> "output.dense.weight")
          <*> lookupTensor (prefix <> "output.dense.bias")
      outProj SBART =
        LinearWithBias
          <$> lookupTensor (prefix <> "out_proj.weight")
          <*> lookupTensor (prefix <> "out_proj.bias")
      outProj SPegasus =
        LinearWithBias
          <$> lookupTensor (prefix <> "out_proj.weight")
          <*> lookupTensor (prefix <> "out_proj.bias")
      dropout _ = pure (initialize @(Dropout dropoutP) dropoutP)
   in MultiHeadAttention
        <$> ( GMultiHeadAttention
                <$> pure headDim
                <*> pure headEmbedDim
                <*> pure embedDim
                <*> qInProj (sing @style)
                <*> kInProj (sing @style)
                <*> vInProj (sing @style)
                <*> outProj (sing @style)
                <*> dropout (sing @style)
            )

-- | Whether or not scaling is applied in the multi-headed attention layer.
data Scaling
  = -- | Scaling is not done.
    NoScaling
  | -- | Scaling is applied to the query after in the in-projection.
    QueryScaling Double
  | -- | Scaling is applied to the attention weights.
    WeightScaling Double

-- | Whether or not out-projection is applied in the multi-headed attention layer.
data OutProj
  = -- | Out-projection is absent.
    NoOutProj
  | -- | Out-projection is applied.
    OutProj

-- | 'HasForward' instance for 'MultiHeadAttention'.
--
-- 'Scaling' of queries is optional.
--
-- @
-- ┌───────────────┐        ┌───────┐       ┌─────┐       ┌───────┐
-- │ attentionBias │        │ query │       │ key │       │ value │
-- └───────┬───────┘        └───┬───┘       └──┬──┘       └───┬───┘
--         │                    │              │              │
--         │                    ▼              ▼              ▼
--         │                mhaQInProj     mhaKInProj     mhaVInProj
--         │                    ▼              │              │
--         │                (scaling)          │              │
--         │                    ▼              ▼              ▼
--         │                 reshape        reshape        reshape
--         │                    ▼              ▼              ▼
--         │                transpose      transpose      transpose
--         │                    │              ▼              │
--         │                    │          transpose          │
--         │                    │              │              │
--         │                    └───►matmul◄───┘              │
--         │                           ▼                      │
--         │                       (scaling)                  │
--         │                           │                      │
--         └──────────►add◄────────────┘                      │
--                      ▼                                     │
--                   softmax                                  │
--                      ▼                                     │
--                  mhaDropout                                │
--                      │                                     │
--                      └──────────────►matmul◄───────────────┘
--                                        ▼
--                                    transpose
--                                        ▼
--                                     reshape
--                                        ▼
--                                   (mhaOutProj)
--                                        │
--                                        ▼
--                                    ┌───────┐
--                                    │ query │
--                                    └───────┘
-- @
instance
  ( SingI style,
    HasForward
      (GMultiHeadAttentionF style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
      ( Scaling,
        Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
        Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
      )
      (Generator generatorDevice)
      output
      generatorOutput,
    output
      ~ Tensor
          'WithGradient
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType <+> valueDataType)
          outputShape,
    generatorOutput
      ~ Generator (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (MultiHeadAttention style device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
    ( Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    (Generator generatorDevice)
    output
    generatorOutput
  where
  forward (MultiHeadAttention gmha@GMultiHeadAttention {..}) (query, key, value, attentionBias) g =
    case sing @style of
      ST5 ->
        forward gmha (NoScaling, query, key, value, attentionBias) g
      SBART ->
        let scaling = 1 / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
         in forward gmha (QueryScaling scaling, query, key, value, attentionBias) g
      SMBART ->
        let scaling = 1 / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
         in forward gmha (QueryScaling scaling, query, key, value, attentionBias) g
      SBERT ->
        let scaling = 1 / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
         in forward gmha (WeightScaling scaling, query, key, value, attentionBias) g
      SRoBERTa ->
        let scaling = 1 / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
         in forward gmha (WeightScaling scaling, query, key, value, attentionBias) g
      SPegasus ->
        let scaling = 1 / (sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ mhaHeadEmbedDim)
         in forward gmha (QueryScaling scaling, query, key, value, attentionBias) g
      _ -> undefined

type BatchDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type BatchDim queryShape keyShape valueShape =
  (queryShape ! 0) <+> (keyShape ! 0) <+> (valueShape ! 0)

getBatchDim ::
  forall m queryShape keyShape valueShape batchDim.
  (MonadFail m, batchDim ~ BatchDim queryShape keyShape valueShape) =>
  SShape queryShape ->
  SShape keyShape ->
  SShape valueShape ->
  m (SDim batchDim)
getBatchDim queryShape keyShape valueShape = do
  queryBatchDim <- sGetDim (SSelectDim $ SByIndex @0) queryShape
  keyBatchDim <- sGetDim (SSelectDim $ SByIndex @0) keyShape
  valueBatchDim <- sGetDim (SSelectDim $ SByIndex @0) valueShape
  keyValueBatchDim <- sUnifyDim keyBatchDim valueBatchDim
  sUnifyDim queryBatchDim keyValueBatchDim

type QuerySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type QuerySeqDim queryShape =
  queryShape ! 1

getQuerySeqDim ::
  forall m queryShape querySeqDim.
  (MonadFail m, querySeqDim ~ QuerySeqDim queryShape) =>
  SShape queryShape ->
  m (SDim querySeqDim)
getQuerySeqDim = sGetDim (SSelectDim $ SByIndex @1)

type KeySeqDim ::
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Shape [Dim (Name Symbol) (Size Nat)] ->
  Dim (Name Symbol) (Size Nat)

type KeySeqDim keyShape valueShape =
  (keyShape ! 1) <+> (valueShape ! 1)

getKeySeqDim ::
  forall m keyShape valueShape keySeqDim.
  (MonadFail m, keySeqDim ~ KeySeqDim keyShape valueShape) =>
  SShape keyShape ->
  SShape valueShape ->
  m (SDim keySeqDim)
getKeySeqDim keyShape valueShape =
  do
    keySeqDim <- sGetDim (SSelectDim $ SByIndex @1) keyShape
    valueSeqDim <- sGetDim (SSelectDim $ SByIndex @1) valueShape
    sUnifyDim keySeqDim valueSeqDim

instance
  ( HasForward
      qInProj
      (Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape)
      generator
      (Tensor qRequiresGradient qLayout qDevice qDataType qShape0)
      qGeneratorOutput,
    qShape
      ~ TransposeF
          ('SelectDim ('ByIndex 1))
          ('SelectDim ('ByIndex 2))
          ( ReshapeF
              qShape0
              ('Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
          ),
    HasForward
      kInProj
      (Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape)
      qGeneratorOutput
      (Tensor qRequiresGradient kLayout kDevice kDataType kShape0)
      kGeneratorOutput,
    weightsShape0
      ~ SoftmaxF
          ('SelectDim ('ByIndex 3))
          ( BroadcastShapesF
              ( MatmulF
                  qShape
                  ( TransposeF
                      ('SelectDim ('ByIndex 2))
                      ('SelectDim ('ByIndex 3))
                      ( TransposeF
                          ('SelectDim ('ByIndex 1))
                          ('SelectDim ('ByIndex 2))
                          ( ReshapeF
                              kShape0
                              ('Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                          )
                      )
                  )
              )
              attentionBiasShape
          ),
    HasForward
      dropout
      ( Tensor
          (qRequiresGradient <|> attentionBiasRequiresGradient)
          (qLayout <+> kLayout <+> attentionBiasLayout)
          (qDevice <+> kDevice <+> attentionBiasDevice)
          (qDataType <+> kDataType <+> attentionBiasDataType)
          weightsShape0
      )
      kGeneratorOutput
      (Tensor weightsRequiresGradient weightsLayout weightsDevice weightsDataType weightsShape)
      weightsGeneratorOutput,
    HasForward
      vInProj
      (Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape)
      weightsGeneratorOutput
      (Tensor weightsRequiresGradient vLayout vDevice vDataType vShape0)
      vGeneratorOutput,
    outputQueryShape0
      ~ TransposeF
          ('SelectDim ('ByIndex 1))
          ('SelectDim ('ByIndex 2))
          ( MatmulF
              weightsShape
              ( TransposeF
                  ('SelectDim ('ByIndex 1))
                  ('SelectDim ('ByIndex 2))
                  ( ReshapeF
                      vShape0
                      ('Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                  )
              )
          ),
    HasForward
      outProj
      ( Tensor
          weightsRequiresGradient
          (weightsLayout <+> vLayout)
          (weightsDevice <+> vDevice)
          (weightsDataType <+> vDataType)
          (ReshapeF outputQueryShape0 ('Shape '[batchDim, querySeqDim, embedDim]))
      )
      vGeneratorOutput
      output
      generatorOutput,
    KnownDim batchDim,
    KnownDim querySeqDim,
    KnownDim keySeqDim,
    KnownDim embedDim,
    SGetShape queryShape,
    SGetShape keyShape,
    SGetShape valueShape,
    batchDim ~ BatchDim queryShape keyShape valueShape,
    querySeqDim ~ QuerySeqDim queryShape,
    keySeqDim ~ KeySeqDim keyShape valueShape
  ) =>
  HasForward
    (GMultiHeadAttention headDim headEmbedDim embedDim qInProj kInProj vInProj outProj dropout)
    ( Scaling,
      Tensor queryRequiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor keyRequiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor valueRequiresGradient valueLayout valueDevice valueDataType valueShape,
      Tensor attentionBiasRequiresGradient attentionBiasLayout attentionBiasDevice attentionBiasDataType attentionBiasShape
    )
    generator
    output
    generatorOutput
  where
  forward GMultiHeadAttention {..} (scaling, query, key, value, attentionBias) =
    runIxState $
      let batchDim = unsafePerformIO $ do
            queryShape <- sShape query
            keyShape <- sShape key
            valueShape <- sShape value
            getBatchDim queryShape keyShape valueShape
          querySeqDim = unsafePerformIO $ do
            queryShape <- sShape query
            getQuerySeqDim queryShape
          keySeqDim = unsafePerformIO $ do
            keyShape <- sShape key
            valueShape <- sShape value
            getKeySeqDim keyShape valueShape
          q =
            ireturn query
              >>>= IxState . forward mhaQInProj
              >>>= ireturn
                . ( \case
                      NoScaling -> id
                      QueryScaling s -> flip mulScalar s
                      WeightScaling _ -> id
                  )
                  scaling
              >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
          k =
            ireturn key
              >>>= IxState . forward mhaKInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
          kt = k >>>= ireturn . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
          weights =
            matmul <<$>> q <<*>> kt
              >>>= ireturn
                . ( \case
                      NoScaling -> id
                      QueryScaling _ -> id
                      WeightScaling s -> flip mulScalar s
                  )
                  scaling
              >>>= ireturn . (`add` attentionBias)
              >>>= IxState . forward mhaDropout . softmax (SSelectDim $ SByIndex @3)
          v =
            ireturn value
              >>>= IxState . forward mhaVInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
       in matmul <<$>> weights <<*>> v
            >>>= ireturn . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
            >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaEmbedDim :|: SNil)
            >>>= IxState . forward mhaOutProj

testMHA ::
  IO
    ( Tensor
        'WithGradient
        ('Layout 'Dense)
        ('Device 'CPU)
        ('DataType 'Float)
        ( 'Shape
            '[ 'Dim ('Name "*") ('Size 1),
               'Dim ('Name "*") ('Size 4),
               'Dim ('Name "*") ('Size 3)
             ]
        )
    )
testMHA = do
  let q = LinearWithoutBias (ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 3)]))
      k = LinearWithoutBias (ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 3)]))
      v = LinearWithoutBias (ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "*") ('Size 2), 'Dim ('Name "*") ('Size 3)]))
      o = LinearWithoutBias (ones @'WithGradient @('Layout 'Dense) @('Device 'CPU) @('DataType 'Float) @('Shape '[ 'Dim ('Name "*") ('Size 3), 'Dim ('Name "*") ('Size 2)]))
      dropout = Dropout 0.1
      mha =
        MultiHeadAttention
          @'T5
          @('Device 'CPU)
          @('DataType 'Float)
          @('Dim ('Name "*") ('Size 1))
          @('Dim ('Name "*") ('Size 2))
          @('Dim ('Name "*") ('Size 2))
          @('Dim ('Name "*") ('Size 3))
          @('Dim ('Name "*") ('Size 3))
          @('Dim ('Name "*") ('Size 3))
          @Float
          ( GMultiHeadAttention
              (SDim (SName @"*") (SSize @1))
              (SDim (SName @"*") (SSize @2))
              (SDim (SName @"*") (SSize @2))
              q
              k
              v
              o
              dropout
          )
  query <-
    case Torch.Tensor.asTensor [[[0 :: Float, 1, 2], [-1, -2, -3], [7, -2, -3], [-1, 5, -3]]] of
      Torch.Tensor.Unsafe t ->
        pure (UnsafeTensor @'WithoutGradient t)
          >>= checkedLayout @('Layout 'Dense)
          >>= checkedDevice @('Device 'CPU)
          >>= checkedDataType @('DataType 'Float)
          >>= checkedShape @('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 4), 'Dim ('Name "*") ('Size 3)])
  key <-
    case Torch.Tensor.asTensor [[[0 :: Float, 0.5, 1], [-0.1, -0.2, -0.3], [-1, 0, 1]]] of
      Torch.Tensor.Unsafe t ->
        pure (UnsafeTensor @'WithoutGradient t)
          >>= checkedLayout @('Layout 'Dense)
          >>= checkedDevice @('Device 'CPU)
          >>= checkedDataType @('DataType 'Float)
          >>= checkedShape @('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 3), 'Dim ('Name "*") ('Size 3)])
  let value = key
  attentionBias <-
    case Torch.Tensor.asTensor
      [ [ [ [0 :: Float, 3, 3],
            [1, 0, 3],
            [1, 1, 0],
            [1, 1, 1]
          ]
        ]
      ] of
      Torch.Tensor.Unsafe t ->
        pure (UnsafeTensor @'WithoutGradient t)
          >>= checkedLayout @('Layout 'Dense)
          >>= checkedDevice @('Device 'CPU)
          >>= checkedDataType @('DataType 'Float)
          >>= checkedShape @('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 4), 'Dim ('Name "*") ('Size 3)])
  g <- mkGenerator @('Device 'CPU) 0
  let (output, _) = forward mha (query, key, value, attentionBias) g
  case output of
    UnsafeTensor t ->
      print (Torch.Tensor.Unsafe t)
  pure output
