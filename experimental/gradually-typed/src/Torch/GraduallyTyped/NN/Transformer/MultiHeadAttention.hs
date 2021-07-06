{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
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

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..), IxStateT (..))
import Control.Monad.State (evalStateT)
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import qualified Data.Map.Strict as Map
import Data.Singletons (SingI (..), SingKind (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (..))
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator, sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient, RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, sGetDim, sUnifyDim, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, sReshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, mulScalar)
import Torch.GraduallyTyped.Tensor.Type (SGetShape (..), Tensor (..))
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Control.Monad.Indexed.Trans (IxMonadTrans(ilift))

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
    (gradient :: Gradient RequiresGradient)
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
    forall style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
    GMultiHeadAttention
      headDim
      headEmbedDim
      embedDim
      (QInProjF style gradient device dataType queryEmbedDim embedDim)
      (KInProjF style gradient device dataType keyEmbedDim embedDim)
      (VInProjF style gradient device dataType valueEmbedDim embedDim)
      (OutProjF style gradient device dataType embedDim queryEmbedDim)
      (DropoutF style dropoutP) ->
    MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type family
  QInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  QInProjF 'T5 gradient device dataType queryEmbedDim embedDim = Linear 'WithoutBias gradient device dataType queryEmbedDim embedDim
  QInProjF 'ByT5 gradient device dataType queryEmbedDim embedDim = QInProjF 'T5 gradient device dataType queryEmbedDim embedDim
  QInProjF _ gradient device dataType queryEmbedDim embedDim = Linear 'WithBias gradient device dataType queryEmbedDim embedDim

type family
  KInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  KInProjF 'T5 gradient device dataType keyEmbedDim embedDim = Linear 'WithoutBias gradient device dataType keyEmbedDim embedDim
  KInProjF 'ByT5 gradient device dataType keyEmbedDim embedDim = KInProjF 'T5 gradient device dataType keyEmbedDim embedDim
  KInProjF _ gradient device dataType keyEmbedDim embedDim = Linear 'WithBias gradient device dataType keyEmbedDim embedDim

type family
  VInProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  VInProjF 'T5 gradient device dataType valueEmbedDim embedDim = Linear 'WithoutBias gradient device dataType valueEmbedDim embedDim
  VInProjF 'ByT5 gradient device dataType valueEmbedDim embedDim = VInProjF 'T5 gradient device dataType valueEmbedDim embedDim
  VInProjF _ gradient device dataType valueEmbedDim embedDim = Linear 'WithBias gradient device dataType valueEmbedDim embedDim

type family
  OutProjF
    (style :: TransformerStyle)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  OutProjF 'T5 gradient device dataType embedDim queryEmbedDim = Linear 'WithoutBias gradient device dataType embedDim queryEmbedDim
  OutProjF 'ByT5 gradient device dataType embedDim queryEmbedDim = OutProjF 'T5 gradient device dataType embedDim queryEmbedDim
  OutProjF _ gradient device dataType embedDim queryEmbedDim = Linear 'WithBias gradient device dataType embedDim queryEmbedDim

type family
  DropoutF
    (style :: TransformerStyle)
    (dropoutP :: Type) ::
    Type
  where
  DropoutF _ dropoutP = Dropout dropoutP

instance
  ( qInProj ~ QInProjF style gradient device dataType queryEmbedDim embedDim,
    HasInitialize qInProj (SGradient gradient, SDevice device, SDataType dataType, SDim queryEmbedDim, SDim embedDim) generator generator',
    kInProj ~ KInProjF style gradient device dataType keyEmbedDim embedDim,
    HasInitialize kInProj (SGradient gradient, SDevice device, SDataType dataType, SDim keyEmbedDim, SDim embedDim) generator' generator'',
    vInProj ~ VInProjF style gradient device dataType valueEmbedDim embedDim,
    HasInitialize vInProj (SGradient gradient, SDevice device, SDataType dataType, SDim valueEmbedDim, SDim embedDim) generator'' generator''',
    outProj ~ OutProjF style gradient device dataType embedDim queryEmbedDim,
    HasInitialize outProj (SGradient gradient, SDevice device, SDataType dataType, SDim embedDim, SDim queryEmbedDim) generator''' generator'''',
    dropout ~ DropoutF style dropoutP,
    HasInitialize dropout dropoutP generator'''' generator''''
  ) =>
  HasInitialize
    (MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, SDim valueEmbedDim, dropoutP)
    generator
    generator''''
  where
  initialize (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, valueEmbedDim, dropoutP) =
    let qInProj = IxState $ initialize (gradient, device, dataType, queryEmbedDim, embedDim)
        kInProj = IxState $ initialize (gradient, device, dataType, keyEmbedDim, embedDim)
        vInProj = IxState $ initialize (gradient, device, dataType, valueEmbedDim, embedDim)
        outProj = IxState $ initialize (gradient, device, dataType, embedDim, queryEmbedDim)
        dropout = IxState $ initialize dropoutP
     in runIxState $
          ( GMultiHeadAttention
              <<$>> ireturn headDim
              <<*>> ireturn headEmbedDim
              <<*>> ireturn embedDim
              <<*>> qInProj
              <<*>> kInProj
              <<*>> vInProj
              <<*>> outProj
              <<*>> dropout
          )
            >>>= ireturn . MultiHeadAttention

instance
  SingI style =>
  HasStateDict
    (MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
    (SGradient gradient, SDevice device, SDataType dataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim queryEmbedDim, SDim keyEmbedDim, SDim valueEmbedDim, dropoutP)
  where
  fromStateDict (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, valueEmbedDim, dropoutP) k =
    let qInProj ST5 = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "q.")
        qInProj SByT5 = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "q.")
        qInProj SBART = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "q_proj.")
        qInProj SMBART = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "q_proj.")
        qInProj SPegasus = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "q_proj.")
        qInProj SBERT = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "self.query.")
        qInProj SRoBERTa = fromStateDict (gradient, device, dataType, queryEmbedDim, embedDim) (k <> "self.query.")
        qInProj SGPT2 = undefined
        kInProj ST5 = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "k.")
        kInProj SByT5 = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "k.")
        kInProj SBART = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "k_proj.")
        kInProj SMBART = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "k_proj.")
        kInProj SPegasus = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "k_proj.")
        kInProj SBERT = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "self.key.")
        kInProj SRoBERTa = fromStateDict (gradient, device, dataType, keyEmbedDim, embedDim) (k <> "self.key.")
        kInProj SGPT2 = undefined
        vInProj ST5 = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "v.")
        vInProj SByT5 = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "v.")
        vInProj SBART = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "v_proj.")
        vInProj SMBART = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "v_proj.")
        vInProj SPegasus = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "v_proj.")
        vInProj SBERT = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "self.value.")
        vInProj SRoBERTa = fromStateDict (gradient, device, dataType, valueEmbedDim, embedDim) (k <> "self.value.")
        vInProj SGPT2 = undefined
        outProj ST5 = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "o.")
        outProj SByT5 = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "o.")
        outProj SBART = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "out_proj.")
        outProj SMBART = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "out_proj.")
        outProj SPegasus = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "out_proj.")
        outProj SBERT = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "output.dense.")
        outProj SRoBERTa = fromStateDict (gradient, device, dataType, embedDim, queryEmbedDim) (k <> "output.dense.")
        outProj SGPT2 = undefined
        dropout _ = fromStateDict dropoutP k
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
  toStateDict k (MultiHeadAttention GMultiHeadAttention {..}) =
    let qInProj ST5 = toStateDict (k <> "q.")
        qInProj SByT5 = toStateDict (k <> "q.")
        qInProj SBART = toStateDict (k <> "q_proj.")
        qInProj SMBART = toStateDict (k <> "q_proj.")
        qInProj SPegasus = toStateDict (k <> "q_proj.")
        qInProj SBERT = toStateDict (k <> "self.query.")
        qInProj SRoBERTa = toStateDict (k <> "self.query.")
        qInProj SGPT2 = undefined
        kInProj ST5 = toStateDict (k <> "k.")
        kInProj SByT5 = toStateDict (k <> "k.")
        kInProj SBART = toStateDict (k <> "k_proj.")
        kInProj SMBART = toStateDict (k <> "k_proj.")
        kInProj SPegasus = toStateDict (k <> "k_proj.")
        kInProj SBERT = toStateDict (k <> "self.key.")
        kInProj SRoBERTa = toStateDict (k <> "self.key.")
        kInProj SGPT2 = undefined
        vInProj ST5 = toStateDict (k <> "v.")
        vInProj SByT5 = toStateDict (k <> "v.")
        vInProj SBART = toStateDict (k <> "v_proj.")
        vInProj SMBART = toStateDict (k <> "v_proj.")
        vInProj SPegasus = toStateDict (k <> "v_proj.")
        vInProj SBERT = toStateDict (k <> "self.value.")
        vInProj SRoBERTa = toStateDict (k <> "self.value.")
        vInProj SGPT2 = undefined
        outProj ST5 = toStateDict (k <> "o.")
        outProj SByT5 = toStateDict (k <> "o.")
        outProj SBART = toStateDict (k <> "out_proj.")
        outProj SMBART = toStateDict (k <> "out_proj.")
        outProj SPegasus = toStateDict (k <> "out_proj.")
        outProj SBERT = toStateDict (k <> "output.dense.")
        outProj SRoBERTa = toStateDict (k <> "output.dense.")
        outProj SGPT2 = undefined
     in do
          () <- qInProj (sing @style) mhaQInProj
          () <- kInProj (sing @style) mhaKInProj
          () <- vInProj (sing @style) mhaVInProj
          () <- outProj (sing @style) mhaOutProj
          pure ()

-- | Whether or not scaling is applied in the multi-headed attention layer.
data Scaling
  = -- | Scaling is not done.
    NoScaling
  | -- | Scaling is applied to the query after in the in-projection.
    QueryScaling Double
  | -- | Scaling is applied to the attention weights.
    WeightScaling Double
  deriving stock (Eq, Ord, Show, Generic)

-- | Whether or not out-projection is applied in the multi-headed attention layer.
data OutProj
  = -- | Out-projection is absent.
    NoOutProj
  | -- | Out-projection is applied.
    OutProj
  deriving stock (Eq, Ord, Show, Generic)

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
      ( GMultiHeadAttention
          headDim
          headEmbedDim
          embedDim
          (QInProjF style gradient device dataType queryEmbedDim embedDim)
          (KInProjF style gradient device dataType keyEmbedDim embedDim)
          (VInProjF style gradient device dataType valueEmbedDim embedDim)
          (OutProjF style gradient device dataType embedDim queryEmbedDim)
          (DropoutF style dropoutP)
      )
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
          gradient
          ('Layout 'Dense <+> queryLayout <+> keyLayout <+> attentionBiasLayout <+> valueLayout)
          (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice <+> valueDevice)
          (dataType <+> queryDataType <+> keyDataType <+> attentionBiasDataType <+> valueDataType)
          outputShape,
    generatorOutput
      ~ Generator (device <+> queryDevice <+> keyDevice <+> attentionBiasDevice <+> generatorDevice)
  ) =>
  HasForward
    (MultiHeadAttention style gradient device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
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
      SByT5 ->
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
  (MonadThrow m, batchDim ~ BatchDim queryShape keyShape valueShape) =>
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
  (MonadThrow m, querySeqDim ~ QuerySeqDim queryShape) =>
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
  (MonadThrow m, keySeqDim ~ KeySeqDim keyShape valueShape) =>
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
  forward GMultiHeadAttention {..} (scaling, query, key, value, attentionBias) g = do
    batchDim <-
      let queryShape = sShape query
          keyShape = sShape key
          valueShape = sShape value
       in getBatchDim queryShape keyShape valueShape
    querySeqDim <-
      let queryShape = sShape query
       in getQuerySeqDim queryShape
    keySeqDim <-
      let keyShape = sShape key
          valueShape = sShape value
       in getKeySeqDim keyShape valueShape
    flip runIxStateT g $
      let q =
            ireturn query
              >>>= IxStateT . forward mhaQInProj
              >>>= ireturn
                . ( \case
                      NoScaling -> id
                      QueryScaling s -> flip mulScalar s
                      WeightScaling _ -> id
                  )
                  scaling
              >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
          k =
            ireturn key
              >>>= IxStateT . forward mhaKInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
          kt = k >>>= ilift . transpose @('SelectDim ('ByIndex 2)) @('SelectDim ('ByIndex 3))
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
              >>>= IxStateT . forward mhaDropout . softmax (SSelectDim $ SByIndex @3)
          v =
            ireturn value
              >>>= IxStateT . forward mhaVInProj
              >>>= ireturn . sReshape (SShape $ batchDim :|: keySeqDim :|: mhaHeadDim :|: mhaHeadEmbedDim :|: SNil)
              >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
       in matmul <<$>> weights <<*>> v
            >>>= ilift . transpose @('SelectDim ('ByIndex 1)) @('SelectDim ('ByIndex 2))
            >>>= ireturn . sReshape (SShape $ batchDim :|: querySeqDim :|: mhaEmbedDim :|: SNil)
            >>>= IxStateT . forward mhaOutProj

testMHA = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      dataType = SDataType SFloat
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      queryEmbedDim = SName @"*" :&: SSize @512
      keyEmbedDim = queryEmbedDim
      valueEmbedDim = queryEmbedDim
      dropoutP :: Float = 0.0
  g <- sMkGenerator device 0
  let (mha, g') = initialize @(MultiHeadAttention 'T5 _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, valueEmbedDim, dropoutP) g
  mha' <- flip evalStateT Map.empty $ do
    toStateDict "mha" mha
    fromStateDict @(MultiHeadAttention 'T5 _ _ _ _ _ _ _ _ _ _) (gradient, device, dataType, headDim, headEmbedDim, embedDim, queryEmbedDim, keyEmbedDim, valueEmbedDim, dropoutP) "mha"
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @4
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      query = sOnes' dataType (SShape $ batchDim :|: seqDim :|: queryEmbedDim :|: SNil)
      key = sOnes' dataType (SShape $ batchDim :|: seqDim :|: keyEmbedDim :|: SNil)
      value = sOnes' dataType (SShape $ batchDim :|: seqDim :|: valueEmbedDim :|: SNil)
      attentionBias = sOnes' dataType (SShape $ batchDim :|: SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
  (output, _) <- forward mha' (query, key, value, attentionBias) g'
  pure output