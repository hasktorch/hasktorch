{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
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

module Torch.GraduallyTyped.NN.Transformer.SequenceToSequence where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader)
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Singletons (SingI, SingKind (fromSing), sing)
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, SDataType)
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder, lookupDecoder)
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder, lookupEncoder)
import Torch.GraduallyTyped.NN.Transformer.LMHead (LMHead, lookupLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (STransformerStyle (..), TensorDict, TransformerStyle (..), lookupTensor)
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (mulScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor (UnsafeTensor))
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Tensor as Tensor

data
  GSequenceToSequenceTransformer
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoder :: Type)
    (decoder :: Type)
    (embedding :: Type)
  where
  GSequenceToSequenceTransformer ::
    forall inputEmbedDim encoder decoder embedding.
    { -- | input embedding dim for scaling
      seqToSeqInputEmbedDim :: SDim inputEmbedDim,
      -- | encoder
      seqToSeqEncoder :: encoder,
      -- | decoder
      seqToSeqDecoder :: decoder,
      -- | shared embedding
      seqToSeqEmbedding :: embedding
    } ->
    GSequenceToSequenceTransformer inputEmbedDim encoder decoder embedding

-- | Sequence-to-sequence transformer model.
newtype
  SequenceToSequenceTransformer
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformer ::
    forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP.
    GSequenceToSequenceTransformerF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP ->
    SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

type GSequenceToSequenceTransformerF
  (numEncoderLayers :: Nat)
  (numDecoderLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (vocabDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GSequenceToSequenceTransformer
    inputEmbedDim
    (SeqToSeqEncoderF numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (SeqToSeqDecoderF numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
    (SeqToSeqEmbeddingF style device dataType inputEmbedDim vocabDim)

type family
  SeqToSeqEncoderF
    (numEncoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SeqToSeqEncoderF numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP = TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP

type family
  SeqToSeqDecoderF
    (numEncoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SeqToSeqDecoderF numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP = TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP

type family
  SeqToSeqEmbeddingF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  SeqToSeqEmbeddingF _ device dataType inputEmbedDim vocabDim = Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing

data
  GSequenceToSequenceTransformerWithLMHead
    (transformer :: Type)
    (lmHead :: Type)
  where
  GSequenceToSequenceTransformerWithLMHead ::
    forall transformer lmHead.
    { -- | sequence-to-sequence transformer
      seqToSeqTransformer :: transformer,
      -- | language modelling head
      seqToSeqLMHead :: lmHead
    } ->
    GSequenceToSequenceTransformerWithLMHead transformer lmHead

-- | Sequence-to-sequence transformer model with language modelling head.
data
  SequenceToSequenceTransformerWithLMHead
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformerWithLMHead ::
    forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP.
    GSequenceToSequenceTransformerWithLMHeadF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP ->
    SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

type GSequenceToSequenceTransformerWithLMHeadF
  (numEncoderLayers :: Nat)
  (numDecoderLayers :: Nat)
  (style :: TransformerStyle)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (vocabDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  GSequenceToSequenceTransformerWithLMHead
    (SeqToSeqTransformerF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SeqToSeqLMHeadF style device dataType inputEmbedDim vocabDim)

type family
  SeqToSeqTransformerF
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Type
  where
  SeqToSeqTransformerF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP = SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

type family
  SeqToSeqLMHeadF
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SeqToSeqLMHeadF style device dataType inputEmbedDim vocabDim = LMHead style device dataType inputEmbedDim vocabDim

instance
  ( HasInitialize (TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP),
    HasInitialize (TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
  ) =>
  HasInitialize (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim inputEmbedDim ->
      SDim ffnDim ->
      SDim posEncDim ->
      SDim vocabDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps = runState $ do
    encoder <-
      state $
        initialize @(TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
          device
          dataType
          headDim
          headEmbedDim
          embedDim
          inputEmbedDim
          ffnDim
          posEncDim
          dropoutP
          eps
    decoder <-
      state $
        initialize @(TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
          device
          dataType
          headDim
          headEmbedDim
          embedDim
          inputEmbedDim
          inputEmbedDim
          ffnDim
          posEncDim
          dropoutP
          eps
    embedding <-
      state $
        initialize @(Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (SLayout SDense)
          device
          dataType
          vocabDim
          inputEmbedDim
    pure . SequenceToSequenceTransformer $ GSequenceToSequenceTransformer inputEmbedDim encoder decoder embedding

instance
  ( SingI style,
    seqToSeqTransformer ~ SeqToSeqTransformerF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP,
    HasInitialize seqToSeqTransformer,
    seqToSeqLMHead ~ SeqToSeqLMHeadF style device dataType inputEmbedDim vocabDim,
    HasInitialize seqToSeqLMHead
  ) =>
  HasInitialize (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) =
      SDevice device ->
      SDataType dataType ->
      SDim headDim ->
      SDim headEmbedDim ->
      SDim embedDim ->
      SDim inputEmbedDim ->
      SDim ffnDim ->
      SDim posEncDim ->
      SDim vocabDim ->
      dropoutP ->
      Double ->
      Generator device ->
      (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device)
  initialize device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps =
    runState $ do
      transformer <- state $ initialize @seqToSeqTransformer device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps
      lmHead <- state $ initialize @seqToSeqLMHead device dataType inputEmbedDim vocabDim eps
      pure . SequenceToSequenceTransformerWithLMHead $ GSequenceToSequenceTransformerWithLMHead transformer lmHead

lookupSequenceToSequenceTransformer ::
  forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    Scalar dropoutP,
    HasLookupStack numEncoderLayers (1 <=? numEncoderLayers) numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m,
    HasLookupDecoderStack numDecoderLayers (1 <=? numDecoderLayers) numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim inputEmbedDim ->
  dropoutP ->
  Double ->
  String ->
  m (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
lookupSequenceToSequenceTransformer headDim headEmbedDim embedDim inputEmbedDim dropoutP eps prefix =
  let encoder ST5 = lookupEncoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "encoder.")
      encoder SBART = lookupEncoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "encoder.")
      encoder SPegasus = lookupEncoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "encoder.")
      decoder ST5 = lookupDecoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "decoder.")
      decoder SBART = lookupDecoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "decoder.")
      decoder SPegasus = lookupDecoder headDim headEmbedDim embedDim dropoutP eps (prefix <> "decoder.")
      embedding ST5 = Embedding <$> lookupTensor "shared.weight"
      embedding SBART = Embedding <$> lookupTensor (prefix <> "shared.weight")
      embedding SPegasus = Embedding <$> lookupTensor (prefix <> "shared.weight")
   in SequenceToSequenceTransformer
        <$> ( GSequenceToSequenceTransformer
                <$> pure inputEmbedDim
                <*> encoder (sing @style)
                <*> decoder (sing @style)
                <*> embedding (sing @style)
            )

lookupSequenceToSequenceTransformerWithLMHead ::
  forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP m.
  ( SingI style,
    MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownDevice device,
    KnownDataType dataType,
    KnownDim headDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim posEncDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    Scalar dropoutP,
    HasLookupStack numEncoderLayers (1 <=? numEncoderLayers) numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP m,
    HasLookupDecoderStack numDecoderLayers (1 <=? numDecoderLayers) numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim dropoutP m
  ) =>
  SDim headDim ->
  SDim headEmbedDim ->
  SDim embedDim ->
  SDim inputEmbedDim ->
  dropoutP ->
  Double ->
  String ->
  m (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
lookupSequenceToSequenceTransformerWithLMHead headDim headEmbedDim embedDim inputEmbedDim dropoutP eps prefix =
  let transformer ST5 = lookupSequenceToSequenceTransformer headDim headEmbedDim embedDim inputEmbedDim dropoutP eps prefix
      transformer SBART = lookupSequenceToSequenceTransformer headDim headEmbedDim embedDim inputEmbedDim dropoutP eps (prefix <> "model.")
      transformer SPegasus = lookupSequenceToSequenceTransformer headDim headEmbedDim embedDim inputEmbedDim dropoutP eps (prefix <> "model.")
      lmHead ST5 = lookupLMHead inputEmbedDim eps (prefix <> "lm_head.")
      lmHead SBART = lookupLMHead inputEmbedDim eps prefix
      lmHead SPegasus = lookupLMHead inputEmbedDim eps prefix
   in SequenceToSequenceTransformerWithLMHead
        <$> ( GSequenceToSequenceTransformerWithLMHead
                <$> transformer (sing @style)
                <*> lmHead (sing @style)
            )

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for training.
data SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerInput ::
    forall input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask.
    { input :: input,
      decoderInput :: decoderInput,
      pos :: pos,
      decoderPos :: decoderPos,
      attentionMask :: attentionMask,
      decoderAttentionMask :: decoderAttentionMask,
      crossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask

deriving instance
  ( Show input,
    Show decoderInput,
    Show pos,
    Show decoderPos,
    Show attentionMask,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)

-- | Output data type for use with a sequence-to-sequence transformer.
data SequenceToSequenceTransformerOutput decoderOutput encoderOutput where
  SequenceToSequenceTransformerOutput ::
    forall decoderOutput encoderOutput.
    { decoderOutput :: decoderOutput,
      encoderOutput :: encoderOutput
    } ->
    SequenceToSequenceTransformerOutput decoderOutput encoderOutput

deriving instance
  ( Show decoderOutput,
    Show encoderOutput
  ) =>
  Show (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for inference.
data SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerGenerationInput ::
    forall decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask.
    { generationDecoderInput :: decoderInput,
      generationEncoderOutput :: encoderOutput,
      generationDecoderPos :: decoderPos,
      generationDecoderAttentionMask :: decoderAttentionMask,
      generationCrossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show decoderPos,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)

-- | 'HasForward' instance for sequence-to-sequence transformers without additional head(s).
--
-- @
--     ┌───────┐  ┌─────┐  ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
--     │ input │  │ pos │  │ attentionMask │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
--     └───┬───┘  └──┬──┘  └──────┬────────┘  └──────┬───────┘  └─────┬──────┘  └──────────┬───────────┘  └─────────┬──────────┘
--         │         │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
-- seqToSeqEmbedding │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--   (embedScaling)  │            │                  │                │                    │                        │
--         ▼         │            │                  │                │                    │                        │
--  seqToSeqEncoder◄─┘◄───────────┘                  ▼                │                    │                        │
--         │                                 seqToSeqEmbedding        │                    │                        │
--         │                                         ▼                │                    │                        │
--         │                                   (embedScaling)         │                    │                        │
--         │                                         ▼                │                    │                        │
--         ├─────────────────────────────────►seqToSeqDecoder◄────────┘◄───────────────────┘◄───────────────────────┘
--         │                                         │
--         ▼                                         ▼
-- ┌───────────────┐                         ┌───────────────┐
-- │ encoderOutput │                         │ decoderOutput │
-- └───────────────┘                         └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (SeqToSeqEmbeddingF style device dataType inputEmbedDim vocabDim)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    embeddingOutput ~ Tensor requiresGradient' layout' device' dataType' shape',
    HasForward
      (SeqToSeqEncoderF numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (embeddingOutput, pos, attentionMask)
      embeddingGeneratorOutput
      encoderOutput
      encoderGeneratorOutput,
    HasForward
      (SeqToSeqEmbeddingF style device dataType inputEmbedDim vocabDim)
      decoderInput
      encoderGeneratorOutput
      embeddingOutput'
      embeddingGeneratorOutput',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SeqToSeqDecoderF numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward (SequenceToSequenceTransformer GSequenceToSequenceTransformer {..}) SequenceToSequenceTransformerInput {..} =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ seqToSeqInputEmbedDim
        embedScaling ::
          forall requiresGradient layout device dataType shape.
          STransformerStyle style ->
          Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType shape
        embedScaling ST5 = id
        embedScaling SPegasus = flip mulScalar s
        embedScaling SBART = id
        f (UnsafeTensor t) =
          let t' = Tensor.asValue (Tensor.Unsafe t) :: [[[Float]]]
           in do
                firstBatch <- take 1 t'
                firstPositions <- take 3 firstBatch
                take 3 firstPositions
        g (UnsafeTensor t) =
          let t' = Tensor.asValue (Tensor.Unsafe t) :: [[Int]]
           in do
                firstBatch <- take 1 t'
                take 3 firstBatch
     in runIxState $
          ireturn input
            >>>= IxState . forward seqToSeqEmbedding
            >>>= ireturn . embedScaling (sing @style)
            >>>= (\input' -> IxState $ forward seqToSeqEncoder (input', pos, attentionMask))
            >>>= ( \encoderOutput ->
                     ireturn decoderInput
                       >>>= IxState . forward seqToSeqEmbedding
                       >>>= ireturn . embedScaling (sing @style)
                       >>>= ( \decoderInput' ->
                                IxState $ forward seqToSeqDecoder (decoderInput', encoderOutput, decoderPos, decoderAttentionMask, crossAttentionMask)
                            )
                       >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
                 )

-- | 'HasForward' instance for sequence-to-sequence transformers without language modelling head.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- @
-- ┌───────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐  ┌────────────────────┐
-- │ encoderOutput │  │ decoderInput │  │ decoderPos │  │ decoderAttentionMask │  │ crossAttentionMask │
-- └───────┬───────┘  └───────┬──────┘  └──────┬─────┘  └───────────┬──────────┘  └──────────┬─────────┘
--         │                  │                │                    │                        │
--         │                  ▼                │                    │                        │
--         │          seqToSeqEmbedding        │                    │                        │
--         │                  ▼                │                    │                        │
--         │            (embedScaling)         │                    │                        │
--         │                  ▼                │                    │                        │
--         ├──────────►seqToSeqDecoder◄────────┘◄───────────────────┘◄───────────────────────┘
--         │                  │
--         ▼                  ▼
-- ┌───────────────┐  ┌───────────────┐
-- │ encoderOutput │  │ decoderOutput │
-- └───────────────┘  └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (SeqToSeqEmbeddingF style device dataType inputEmbedDim vocabDim)
      decoderInput
      generator
      embeddingOutput'
      embeddingGeneratorOutput',
    embeddingOutput' ~ Tensor requiresGradient'' layout'' device'' dataType'' shape'',
    HasForward
      (SeqToSeqDecoderF numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward (SequenceToSequenceTransformer GSequenceToSequenceTransformer {..}) SequenceToSequenceTransformerGenerationInput {..} =
    let s :: Double = sqrt . fromIntegral . forgetIsChecked . dimSize . fromSing $ seqToSeqInputEmbedDim
        embedScaling ::
          forall requiresGradient layout device dataType shape.
          STransformerStyle style ->
          Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType shape
        embedScaling ST5 = id
        embedScaling SPegasus = flip mulScalar s
     in runIxState $
          ireturn generationDecoderInput
            >>>= IxState . forward seqToSeqEmbedding
            >>>= ireturn . embedScaling (sing @style)
            >>>= ( \decoderInput' ->
                     IxState $ forward seqToSeqDecoder (decoderInput', generationEncoderOutput, generationDecoderPos, generationDecoderAttentionMask, generationCrossAttentionMask)
                 )
            >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput generationEncoderOutput)

type family
  SequenceToSequenceTransformerWithLMHeadDecoderOutputF
    (style :: TransformerStyle)
    (lmHeadOutput :: Type)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  SequenceToSequenceTransformerWithLMHeadDecoderOutputF 'T5 lmHeadOutput _ _ _ = lmHeadOutput
  SequenceToSequenceTransformerWithLMHeadDecoderOutputF 'Pegasus (Tensor requiresGradient' layout' device' dataType' shape') device dataType vocabDim =
    Tensor
      'WithGradient
      (layout' <+> 'Layout 'Dense)
      (device' <+> device)
      (dataType' <+> dataType)
      (BroadcastShapesF shape' ('Shape '[ 'Dim ('Name "*") ('Size 1), vocabDim]))

-- | 'HasForward' instance for sequence-to-sequence transformers with language modelling head.
--
-- @
--                        ┌───────┐
--                        │ input │
--                        └───┬───┘
--                            │
--                            ▼
--         ┌─────────seqToSeqTransformer
--         │                  ▼
--         │            seqToSeqLMHead
--         │                  │
--         ▼                  ▼
-- ┌───────────────┐  ┌───────────────┐
-- │ encoderOutput │  │ decoderOutput │
-- └───────────────┘  └───────────────┘
-- @
instance
  ( SingI style,
    HasForward
      (SeqToSeqTransformerF numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
      input
      generator
      seqToSeqOutput
      seqToSeqGeneratorOutput,
    seqToSeqOutput ~ SequenceToSequenceTransformerOutput decoderOutput encoderOutput,
    HasForward
      (SeqToSeqLMHeadF style device dataType inputEmbedDim vocabDim)
      decoderOutput
      seqToSeqGeneratorOutput
      lmHeadOutput
      generatorOutput,
    output ~ SequenceToSequenceTransformerOutput lmHeadOutput encoderOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    input
    generator
    output
    generatorOutput
  where
  forward (SequenceToSequenceTransformerWithLMHead GSequenceToSequenceTransformerWithLMHead {..}) input =
    runIxState $
      ireturn input
        >>>= IxState . forward seqToSeqTransformer
        >>>= ( \SequenceToSequenceTransformerOutput {..} ->
                 ireturn decoderOutput
                   >>>= IxState . forward seqToSeqLMHead
                   >>>= \lmHeadOutput -> ireturn (SequenceToSequenceTransformerOutput lmHeadOutput encoderOutput)
             )

testForwardSeqToSeq :: _
testForwardSeqToSeq =
  let seqToSeq =
        undefined ::
          SequenceToSequenceTransformerWithLMHead
            128
            128
            'T5
            ('Device 'CPU)
            ('DataType 'Float)
            ('Dim ('Name "*") ('Size 8)) -- headDim
            ('Dim ('Name "*") ('Size 64)) -- headEmbedDim
            ('Dim ('Name "*") ('Size 512)) -- embedDim
            ('Dim ('Name "*") ('Size 512)) -- inputEmbedDim
            ('Dim ('Name "*") ('Size 2048)) -- ffnDim
            ('Dim ('Name "*") ('Size 32)) -- posEncDim
            ('Dim ('Name "*") ('Size 32128)) -- vocabDim
            Float
      input =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7)])
      decoderInput =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5)])
      pos =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
      decoderPos =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
      attentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
      decoderAttentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
      crossAttentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 7)])
      g = undefined :: Generator ('Device 'CPU)
   in forward seqToSeq (SequenceToSequenceTransformerInput input decoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask) g
