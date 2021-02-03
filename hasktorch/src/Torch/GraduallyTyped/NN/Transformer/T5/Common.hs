{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE IncoherentInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.RewriteRules.LayoutDeviceRule
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.RewriteRules.LayoutDataTypeRule #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Common where

import Control.Monad.Reader (MonadIO, MonadReader, ask, liftIO)
import qualified Data.Map as Map
import Foreign.ForeignPtr (ForeignPtr)
import GHC.Float (double2Int)
import GHC.TypeLits (KnownNat, Nat, Symbol, type (-), type (<=?))
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice)
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock (TransformerBlock))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention (CrossAttention))
import Torch.GraduallyTyped.NN.Transformer.Decoder (TransformerDecoder (TransformerDecoder))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (TransformerDecoderStack (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (TransformerEncoder (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (TransformerFeedForwardNetwork (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (MultiHeadAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (HasLMHead (..), SequenceToSequenceTransformer (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (TransformerStack (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.RewriteRules ()
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), full, ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (logicalOr)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape, device, layout, shape)
import Torch.GraduallyTyped.Unify (type (<+>))
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

-- | T5 dType.
type T5DType = 'Float

-- | T5 data type.
type T5DataType = 'DataType T5DType

-- | T5 dropout probability type.
type T5DropoutP = Float

-- | T5 dropout rate.
-- 'dropout_rate = 0.1'
t5DropoutP :: T5DropoutP
t5DropoutP = 0.1

-- | T5 relative positional encoding bucket dimension.
-- 'relative_attention_num_buckets = 32'
type T5RelPosEncBucketDim = 'Dim ( 'Name "*") ( 'Size 32)

-- | T5 layer-norm epsilon.
-- 'layer_norm_epsilon = 1e-06'
t5Eps :: Double
t5Eps = 1e-6

-- | T5 maximum distance for relative positional encoding.
t5MaxDistance :: Int
t5MaxDistance = 128

-- | T5 padding token id.
-- 'pad_token_id = 0'
t5PadTokenId :: Int
t5PadTokenId = 0

-- | T5 attention mask bias
t5AttentionMaskBias :: Double
t5AttentionMaskBias = -10000

type StateDict = Map.Map String (ForeignPtr ATen.Tensor)

data T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim where
  T5Config ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim.
    ( KnownNat numLayers,
      KnownDevice device,
      KnownDim headDim,
      KnownDim headEmbedDim,
      KnownDim embedDim,
      KnownDim inputEmbedDim,
      KnownDim ffnDim,
      KnownDim relPosEncBucketDim,
      KnownDim vocabDim
    ) =>
    { debug :: Bool,
      dropoutP :: T5DropoutP,
      eps :: Double,
      stateDict :: StateDict
    } ->
    T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim

t5ConfigFromPretrained ::
  (KnownNat numLayers, KnownDim headDim, KnownDim headEmbedDim, KnownDim embedDim, KnownDim inputEmbedDim, KnownDim ffnDim, KnownDim relPosEncBucketDim, KnownDim vocabDim) =>
  FilePath ->
  Bool ->
  IO (T5Config numLayers ( 'Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim)
t5ConfigFromPretrained filePath debug = do
  iValue <- Torch.Serialize.pickleLoad filePath
  stateDict <- case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  pure $ T5Config debug t5DropoutP t5Eps stateDict
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."

class
  HasLookupStack
    (n :: Nat)
    (isCons :: Bool)
    (numLayers :: Nat)
    device
    headDim
    headEmbedDim
    embedDim
    inputEmbedDim
    ffnDim
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    m
  where
  lookupEncoderStack' ::
    Integer ->
    m (TransformerStack n device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5DropoutP)
  lookupDecoderStack' ::
    Integer ->
    m (TransformerDecoderStack n device T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim T5DropoutP)

instance
  Applicative m =>
  HasLookupStack 0 'False numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  where
  lookupEncoderStack' _ = pure TransformerStackNil
  lookupDecoderStack' _ = pure TransformerDecoderStackNil

instance
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack (n - 1) (1 <=? (n - 1)) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  HasLookupStack n 'True numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  where
  lookupEncoderStack' n =
    TransformerStackCons
      <$> lookupEncoderBlock n
      <*> lookupEncoderStack' @(n - 1) @(1 <=? (n - 1)) @numLayers @device @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim (n + 1)
  lookupDecoderStack' n =
    TransformerDecoderStackCons
      <$> lookupDecoderBlock n
      <*> lookupDecoderStack' @(n - 1) @(1 <=? (n - 1)) @numLayers @device @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim (n + 1)

lookupEncoderStack ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m (TransformerStack numLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5DropoutP)
lookupEncoderStack = lookupEncoderStack' @numLayers @(1 <=? numLayers) @numLayers @device @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim 0

lookupDecoderStack ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m (TransformerDecoderStack numLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim T5DropoutP)
lookupDecoderStack = lookupDecoderStack' @numLayers @(1 <=? numLayers) @numLayers @device @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim 0

lookupEncoder ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m
    ( TransformerEncoder
        numLayers
        device
        T5DataType
        headDim
        headEmbedDim
        embedDim
        inputEmbedDim
        ffnDim
        relPosEncBucketDim
        T5DropoutP
    )
lookupEncoder = do
  t5Config <- ask
  case t5Config of
    T5Config {..} ->
      TransformerEncoder
        <$> lookupEncoderStack
        <*> ( LayerNormWithoutBias
                <$> lookupTensor "encoder.final_layer_norm.weight"
                <*> pure eps
            )
        <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
        <*> ( Embedding
                <$> lookupTensor "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            )

lookupDecoder ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m (TransformerDecoder numLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim T5DropoutP)
lookupDecoder = do
  t5Config <- ask
  case t5Config of
    T5Config {..} ->
      TransformerDecoder
        <$> lookupDecoderStack
        <*> ( LayerNormWithoutBias
                <$> lookupTensor "decoder.final_layer_norm.weight"
                <*> pure eps
            )
        <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
        <*> ( Embedding
                <$> lookupTensor "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            )

lookupSequenceToSequenceTransformerWithoutLMHead ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m (SequenceToSequenceTransformer 'WithoutLMHead numLayers numLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim T5DropoutP)
lookupSequenceToSequenceTransformerWithoutLMHead = do
  t5Config <- ask
  case t5Config of
    T5Config {..} ->
      SequenceToSequenceTransformerWithoutLMHead
        <$> lookupEncoder
        <*> lookupDecoder
        <*> ( Embedding
                <$> lookupTensor "shared.weight"
            )

lookupSequenceToSequenceTransformerWithLMHead ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m
  ) =>
  m (SequenceToSequenceTransformer 'WithLMHead numLayers numLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim T5DropoutP)
lookupSequenceToSequenceTransformerWithLMHead = do
  t5Config <- ask
  case t5Config of
    T5Config {..} ->
      SequenceToSequenceTransformerWithLMHead
        <$> lookupEncoder
        <*> lookupDecoder
        <*> ( Embedding
                <$> lookupTensor "shared.weight"
            )
        <*> ( LinearWithoutBias
                <$> lookupTensor "lm_head.weight"
            )
        <*> lookupInputEmbedDim

lookupTensor ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim requiresGradient layout shape m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m,
    KnownLayout layout,
    KnownShape shape
  ) =>
  String ->
  m (Tensor requiresGradient layout device T5DataType shape)
lookupTensor s = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> do
      if debug
        then liftIO . putStrLn $ "loading `" <> s <> "`..."
        else pure ()
      liftIO
        ( maybe
            (fail $ "`" <> show s <> "` is not in the state dictionary.")
            (pure . UnsafeTensor)
            (Map.lookup s stateDict)
        )
        >>= checkedLayout
        >>= checkedDevice
        >>= checkedDataType @T5DataType
        >>= checkedShape

lookupHeadDim ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadFail m
  ) =>
  m (Dim String Integer)
lookupHeadDim = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> case dimVal @embedDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> fail "head dimension unspecified"

lookupHeadEmbedDim ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadFail m
  ) =>
  m (Dim String Integer)
lookupHeadEmbedDim = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> case dimVal @headEmbedDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> fail "head embed dimension unspecified"

lookupInputEmbedDim ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadFail m
  ) =>
  m (Dim String Integer)
lookupInputEmbedDim = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> case dimVal @inputEmbedDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> fail "input embed dimension unspecified"

lookupRelPosEncBucketDim ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadFail m
  ) =>
  m (Dim String Integer)
lookupRelPosEncBucketDim = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> case dimVal @relPosEncBucketDim of
      Dim (Name name) (Size size) -> pure $ Dim name size
      Dim _ _ -> fail "bucket dimension for relative positional encoding unspecified"

lookupEncoderBlock ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m
  ) =>
  Integer ->
  m (TransformerBlock device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5DropoutP)
lookupEncoderBlock n = do
  t5Config <- ask
  case t5Config of
    T5Config {..} -> do
      TransformerBlock
        <$> ( SelfAttention
                <$> ( MultiHeadAttention
                        <$> lookupHeadDim
                        <*> lookupHeadEmbedDim
                        <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.SelfAttention.q.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.SelfAttention.k.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.SelfAttention.v.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.SelfAttention.o.weight"))
                        <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wi.weight"))
                <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wo.weight"))
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
            )

lookupDecoderBlock ::
  forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim m.
  ( MonadReader (T5Config numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim) m,
    MonadIO m,
    MonadFail m
  ) =>
  Integer ->
  m (TransformerDecoderBlock device T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim T5DropoutP)
lookupDecoderBlock n = do
  t5Config <- ask
  case t5Config of
    T5Config {..} ->
      TransformerDecoderBlock
        <$> ( SelfAttention
                <$> ( MultiHeadAttention
                        <$> lookupHeadDim
                        <*> lookupHeadEmbedDim
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.SelfAttention.q.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.SelfAttention.k.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.SelfAttention.v.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.SelfAttention.o.weight"))
                        <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
            )
        <*> ( CrossAttention
                <$> ( MultiHeadAttention
                        <$> lookupHeadDim
                        <*> lookupHeadEmbedDim
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.q.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.k.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.v.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.o.weight"))
                        <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wi.weight"))
                <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wo.weight"))
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout T5DropoutP) dropoutP)
            )

padded :: Integral n => n -> a -> [a] -> [a]
padded n p xs =
  let n' = fromIntegral n
      diff = n' - length xs
   in take n' xs ++ replicate diff p

mkT5Input ::
  forall batchDim seqDim m output.
  ( MonadFail m,
    WithDimC batchDim (WithDimF seqDim ([[Int]] -> m output)),
    WithDimC seqDim ([[Int]] -> m output),
    KnownDim batchDim,
    KnownDim seqDim,
    output
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[batchDim, seqDim])
  ) =>
  WithDimF batchDim (WithDimF seqDim ([[Int]] -> m output))
mkT5Input =
  withDim @batchDim $
    \(Dim batchName batchSize) ->
      withDim @seqDim @([[Int]] -> m output) $
        \(Dim seqName seqSize) xs -> do
          let emptySeq = replicate (fromIntegral seqSize) t5PadTokenId
              paddedXs = padded batchSize emptySeq (padded seqSize t5PadTokenId <$> xs)
          case Torch.Tensor.asTensor paddedXs of
            Torch.Tensor.Unsafe t ->
              pure (UnsafeTensor @ 'WithoutGradient t)
                >>= checkedLayout @( 'Layout 'Dense)
                >>= checkedDevice @( 'Device 'CPU)
                >>= checkedDataType @( 'DataType 'Int64)
                >>= checkedShape @( 'Shape '[batchDim, seqDim])

mkT5PaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ( 'DataType 'Bool))
    (BroadcastShapesF shape ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1)]))
mkT5PaddingMask input =
  let padTokenId = full @ 'WithoutGradient @( 'Layout 'Dense) @( 'Device 'CPU) @( 'DataType 'Int64) @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1)]) t5PadTokenId
   in input ==. padTokenId

mkT5AttentionMask ::
  forall requiresGradient layout device dataType shape seqDim output.
  ( KnownLayout layout,
    KnownDevice device,
    KnownShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          (Seq (requiresGradient <+> 'WithoutGradient) 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) T5DataType)
          ( BroadcastShapesF
              (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) shape)
              ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
          ),
    WithCreateC (Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])) 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  output
mkT5AttentionMask paddingMask =
  let layoutType = layout paddingMask
      deviceType = device paddingMask
      dType = dTypeVal @T5DType
      [_batchDim, seqDim] = shape paddingMask
      emptyMask =
        withoutCreate @(Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])) @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
          (zeros @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim]))
          WithoutGradient
          layoutType
          deviceType
          dType
          [Dim "*" 1, seqDim, seqDim]
   in maskedFill (unsqueeze @( 'SelectDim ( 'ByIndex 1)) paddingMask) t5AttentionMaskBias emptyMask

mkT5DecoderAttentionMask ::
  forall requiresGradient layout device dataType shape seqDim output.
  ( KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          'WithoutGradient
          (layout <+> 'Layout 'Dense)
          device
          T5DataType
          ( BroadcastShapesF
              ( BroadcastShapesF
                  ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
                  (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) shape)
              )
              ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
          ),
    WithCreateC (Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[seqDim, seqDim])) 'WithoutGradient layout device T5DataType ( 'Shape '[seqDim, seqDim]),
    WithCreateC (Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])) 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
  ) =>
  Tensor requiresGradient layout device dataType shape ->
  output
mkT5DecoderAttentionMask paddingMask =
  let layoutType = layout paddingMask
      deviceType = device paddingMask
      dType' = dTypeVal @T5DType
      [_batchDim, seqDim] = shape paddingMask
      causalMask =
        unsqueeze @( 'SelectDim ( 'ByIndex 0))
          . bool
          . triu 1
          $ withoutCreate @(Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[seqDim, seqDim])) @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[seqDim, seqDim])
            (ones @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[seqDim, seqDim]))
            WithoutGradient
            layoutType
            deviceType
            dType'
            [seqDim, seqDim]
      emptyMask =
        withoutCreate @(Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])) @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
          (zeros @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim]))
          WithoutGradient
          layoutType
          deviceType
          dType'
          [Dim "*" 1, seqDim, seqDim]
      booleanMask = causalMask `logicalOr` unsqueeze @( 'SelectDim ( 'ByIndex 1)) paddingMask
   in maskedFill
        booleanMask
        t5AttentionMaskBias
        emptyMask

mkT5CrossAttentionMask ::
  forall seqDim' requiresGradient layout device dataType shape seqDim output.
  ( KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          (Seq (requiresGradient <+> 'WithoutGradient) 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) T5DataType)
          ( BroadcastShapesF
              (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) shape)
              ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim])
          ),
    WithCreateC (Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim])) 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim]),
    WithDimC seqDim' (Tensor requiresGradient layout device dataType shape -> output)
  ) =>
  WithDimF seqDim' (Tensor requiresGradient layout device dataType shape -> output)
mkT5CrossAttentionMask =
  withDim @seqDim' @(Tensor requiresGradient layout device dataType shape -> output) $
    \seqDim' paddingMask ->
      let layoutType = layout paddingMask
          deviceType = device paddingMask
          dType = dTypeVal @T5DType
          [_batchDim, seqDim] = shape paddingMask
          emptyMask =
            withoutCreate @(Tensor 'WithoutGradient layout device T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim])) @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim])
              (zeros @ 'WithoutGradient @layout @device @T5DataType @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim', seqDim]))
              WithoutGradient
              layoutType
              deviceType
              dType
              [Dim "*" 1, seqDim', seqDim]
       in maskedFill (unsqueeze @( 'SelectDim ( 'ByIndex 1)) paddingMask) t5AttentionMaskBias emptyMask

-- >>> mkRelPos' 32 128 21 17
-- [[0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25,26],[1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25,25],[2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25,25],[3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25,25],[4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24,25],[5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24,24],[6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24,24],[7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24,24],[8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23,24],[8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22,23],[8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21,22],[8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20,21],[9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19,20],[9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18,19],[9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17,18],[9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0,17],[10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1,0],[10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2,1],[10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3,2],[10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4,3],[10,10,10,10,10,9,9,9,9,8,8,8,8,7,6,5,4]]
mkT5RelPos' :: Int -> Int -> Int -> Int -> [[Int]]
mkT5RelPos' numBuckets maxDistance querySize keySize =
  let queryPos = [0, 1 .. querySize - 1]
      keyPos = [0, 1 .. keySize - 1]
      numBuckets' = numBuckets `div` 2
      maxExact = numBuckets' `div` 2
   in fmap
        ( \qp ->
            fmap
              ( \kp ->
                  let rawRelPos = kp - qp
                      absRelPos = abs rawRelPos
                      relBucket = if rawRelPos > 0 then numBuckets' else 0
                      relBucket' =
                        let isSmall = absRelPos < maxExact
                            relPosIfLarge =
                              maxExact
                                + double2Int
                                  ( logBase
                                      (fromIntegral maxDistance / fromIntegral maxExact)
                                      (fromIntegral absRelPos / fromIntegral maxExact)
                                      * fromIntegral (numBuckets' - maxExact)
                                  )
                            relPosIfLarge' = min relPosIfLarge (numBuckets' - 1)
                         in if isSmall then absRelPos else relPosIfLarge'
                   in relBucket + relBucket'
              )
              keyPos
        )
        queryPos

mkT5RelPos ::
  forall seqDim (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat)) output.
  ( WithDimC seqDim (WithDimF relPosEncBucketDim (Int -> output)),
    WithDimC relPosEncBucketDim (Int -> output),
    KnownDim seqDim,
    output
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
  ) =>
  WithDimF seqDim (WithDimF relPosEncBucketDim (Int -> output))
mkT5RelPos =
  withDim @seqDim $
    \(Dim seqName seqSize) -> withDim @relPosEncBucketDim @(Int -> output) $
      \(Dim relPosEncBucketName relPosEncBucketSize) maxDistance ->
        let relPosEncBucketSize' = fromIntegral relPosEncBucketSize
            seqSize' = fromIntegral seqSize
         in case Torch.Tensor.asTensor [mkT5RelPos' relPosEncBucketSize' maxDistance seqSize' seqSize'] of
              Torch.Tensor.Unsafe t ->
                unsafePerformIO $
                  pure (UnsafeTensor @ 'WithoutGradient t)
                    >>= checkedLayout @( 'Layout 'Dense)
                    >>= checkedDevice @( 'Device 'CPU)
                    >>= checkedDataType @( 'DataType 'Int64)
                    >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])

-- >>> mkDecoderRelPos' 32 128 21 17
-- [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0],[6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0],[7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0],[8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0],[9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0],[10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0],[11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0],[12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0],[13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0],[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0],[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],[16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1],[16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2],[17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3],[17,17,16,16,16,15,14,13,12,11,10,9,8,7,6,5,4]]
mkT5DecoderRelPos' :: Int -> Int -> Int -> Int -> [[Int]]
mkT5DecoderRelPos' numBuckets maxDistance querySize keySize =
  let queryPos = [0, 1 .. querySize - 1]
      keyPos = [0, 1 .. keySize - 1]
      maxExact = numBuckets `div` 2
   in fmap
        ( \qp ->
            fmap
              ( \kp ->
                  let rawRelPos = kp - qp
                      absRelPos = negate . min 0 $ rawRelPos
                      relBucket' =
                        let isSmall = absRelPos < maxExact
                            relPosIfLarge =
                              maxExact
                                + double2Int
                                  ( logBase
                                      (fromIntegral maxDistance / fromIntegral maxExact)
                                      (fromIntegral absRelPos / fromIntegral maxExact)
                                      * fromIntegral (numBuckets - maxExact)
                                  )
                            relPosIfLarge' = min relPosIfLarge (numBuckets - 1)
                         in if isSmall then absRelPos else relPosIfLarge'
                   in relBucket'
              )
              keyPos
        )
        queryPos

mkT5DecoderRelPos ::
  forall seqDim (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat)) output.
  ( WithDimC seqDim (WithDimF relPosEncBucketDim (Int -> output)),
    WithDimC relPosEncBucketDim (Int -> output),
    KnownDim seqDim,
    output
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])
  ) =>
  WithDimF seqDim (WithDimF relPosEncBucketDim (Int -> output))
mkT5DecoderRelPos =
  withDim @seqDim $
    \(Dim seqName seqSize) -> withDim @relPosEncBucketDim @(Int -> output) $
      \(Dim relPosEncBucketName relPosEncBucketSize) maxDistance ->
        let relPosEncBucketSize' = fromIntegral relPosEncBucketSize
            seqSize' = fromIntegral seqSize
         in case Torch.Tensor.asTensor [mkT5DecoderRelPos' relPosEncBucketSize' maxDistance seqSize' seqSize'] of
              Torch.Tensor.Unsafe t ->
                unsafePerformIO $
                  pure (UnsafeTensor @ 'WithoutGradient t)
                    >>= checkedLayout @( 'Layout 'Dense)
                    >>= checkedDevice @( 'Device 'CPU)
                    >>= checkedDataType @( 'DataType 'Int64)
                    >>= checkedShape @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), seqDim, seqDim])

instance
  ( input
      ~ Tensor
          inputRequiresGradient
          inputLayout
          inputDevice
          inputDataType
          inputShape,
    inputSeqDim ~ (inputShape ! 1),
    KnownDim inputSeqDim,
    KnownShape inputShape,
    decoderInput
      ~ Tensor
          decoderInputRequiresGradient
          decoderInputLayout
          decoderInputDevice
          decoderInputDataType
          decoderInputShape,
    decoderInputSeqDim ~ (decoderInputShape ! 1),
    KnownDim decoderInputSeqDim,
    KnownShape decoderInputShape,
    inputPaddingMask
      ~ Tensor
          inputPaddingMaskRequiresGradient
          inputPaddingMaskLayout
          inputPaddingMaskDevice
          inputPaddingMaskDataType
          inputPaddingMaskShape,
    inputPaddingMaskRequiresGradient ~ 'WithoutGradient,
    inputPaddingMaskLayout ~ (inputLayout <+> 'Layout 'Dense),
    inputPaddingMaskDevice ~ (inputDevice <+> 'Device 'CPU),
    inputPaddingMaskDataType ~ Seq (inputDataType <+> 'DataType 'Int64) ( 'DataType 'Bool),
    inputPaddingMaskShape ~ BroadcastShapesF inputShape ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1)]),
    KnownLayout inputPaddingMaskLayout,
    KnownDevice inputPaddingMaskDevice,
    KnownDataType inputPaddingMaskDataType,
    KnownShape inputPaddingMaskShape,
    decoderInputPaddingMask
      ~ Tensor
          decoderInputPaddingMaskRequiresGradient
          decoderInputPaddingMaskLayout
          decoderInputPaddingMaskDevice
          decoderInputPaddingMaskDataType
          decoderInputPaddingMaskShape,
    decoderInputPaddingMaskRequiresGradient ~ 'WithoutGradient,
    decoderInputPaddingMaskLayout ~ (decoderInputLayout <+> 'Layout 'Dense),
    decoderInputPaddingMaskDevice ~ (decoderInputDevice <+> 'Device 'CPU),
    decoderInputPaddingMaskDataType ~ Seq (decoderInputDataType <+> 'DataType 'Int64) ( 'DataType 'Bool),
    decoderInputPaddingMaskShape ~ BroadcastShapesF decoderInputShape ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1)]),
    KnownLayout decoderInputPaddingMaskLayout,
    KnownDevice decoderInputPaddingMaskDevice,
    KnownDataType decoderInputPaddingMaskDataType,
    KnownShape decoderInputPaddingMaskShape,
    attentionMask
      ~ Tensor
          attentionMaskRequiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          attentionMaskShape,
    attentionMaskRequiresGradient ~ 'WithoutGradient,
    attentionMaskLayout ~ (inputLayout <+> 'Layout 'Dense),
    attentionMaskDevice ~ (inputDevice <+> 'Device 'CPU),
    attentionMaskDataType ~ Seq (inputPaddingMaskDataType <+> 'DataType 'Bool) T5DataType,
    attentionMaskShape
      ~ BroadcastShapesF
          (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) inputPaddingMaskShape)
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), inputPaddingMaskShape ! 1, inputPaddingMaskShape ! 1]),
    WithCreateC (Tensor 'WithoutGradient inputPaddingMaskLayout inputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), inputPaddingMaskShape ! 1, inputPaddingMaskShape ! 1])) 'WithoutGradient inputPaddingMaskLayout inputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), inputPaddingMaskShape ! 1, inputPaddingMaskShape ! 1]),
    decoderAttentionMask
      ~ Tensor
          'WithoutGradient
          (decoderInputLayout <+> 'Layout 'Dense)
          (decoderInputDevice <+> 'Device 'CPU)
          T5DataType
          ( BroadcastShapesF
              ( BroadcastShapesF
                  ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1])
                  (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) decoderInputPaddingMaskShape)
              )
              ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1])
          ),
    WithCreateC (Tensor 'WithoutGradient decoderInputPaddingMaskLayout decoderInputPaddingMaskDevice T5DataType ( 'Shape '[decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1])) 'WithoutGradient decoderInputPaddingMaskLayout decoderInputPaddingMaskDevice T5DataType ( 'Shape '[decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1]),
    WithCreateC (Tensor 'WithoutGradient decoderInputPaddingMaskLayout decoderInputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1])) 'WithoutGradient decoderInputPaddingMaskLayout decoderInputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputPaddingMaskShape ! 1, decoderInputPaddingMaskShape ! 1]),
    crossAttentionMask
      ~ Tensor
          'WithoutGradient
          (inputLayout <+> 'Layout 'Dense)
          (inputDevice <+> 'Device 'CPU)
          (Seq (inputPaddingMaskDataType <+> 'DataType 'Bool) T5DataType)
          ( BroadcastShapesF
              (UnsqueezeF ( 'SelectDim ( 'ByIndex 1)) inputPaddingMaskShape)
              ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputSeqDim, inputPaddingMaskShape ! 1])
          ),
    WithCreateC (Tensor 'WithoutGradient inputPaddingMaskLayout inputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputSeqDim, inputPaddingMaskShape ! 1])) 'WithoutGradient inputPaddingMaskLayout inputPaddingMaskDevice T5DataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputSeqDim, inputPaddingMaskShape ! 1]),
    WithDimC decoderInputSeqDim (inputPaddingMask -> crossAttentionMask),
    relPos
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), inputSeqDim, inputSeqDim]),
    WithDimC inputSeqDim (Int -> relPos),
    decoderRelPos
      ~ Tensor
          'WithoutGradient
          ( 'Layout 'Dense)
          ( 'Device 'CPU)
          ( 'DataType 'Int64)
          ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), decoderInputSeqDim, decoderInputSeqDim]),
    WithDimC decoderInputSeqDim (Int -> decoderRelPos),
    HasForward
      (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim T5DropoutP)
      (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
      generator
      output
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim T5DropoutP)
    ( input,
      decoderInput
    )
    generator
    output
    generatorOutput
  where
  forward seqToSeq (input, decoderInput) =
    let inputPaddingMask = mkT5PaddingMask input
        attentionMask =
          mkT5AttentionMask
            @inputPaddingMaskRequiresGradient
            @inputPaddingMaskLayout
            @inputPaddingMaskDevice
            @inputPaddingMaskDataType
            @inputPaddingMaskShape
            inputPaddingMask
        decoderInputPaddingMask = mkT5PaddingMask decoderInput
        decoderAttentionMask =
          mkT5DecoderAttentionMask
            @decoderInputPaddingMaskRequiresGradient
            @decoderInputPaddingMaskLayout
            @decoderInputPaddingMaskDevice
            @decoderInputPaddingMaskDataType
            @decoderInputPaddingMaskShape
            decoderInputPaddingMask
        [_, decoderInputSeqDim] = shape decoderInput
        crossAttentionMask =
          withoutDim @decoderInputSeqDim @(inputPaddingMask -> crossAttentionMask)
            ( mkT5CrossAttentionMask
                @decoderInputSeqDim
                @inputPaddingMaskRequiresGradient
                @inputPaddingMaskLayout
                @inputPaddingMaskDevice
                @inputPaddingMaskDataType
                @inputPaddingMaskShape
            )
            decoderInputSeqDim
            inputPaddingMask
        [_, inputSeqDim] = shape input
        relPos =
          withoutDim @inputSeqDim @(Int -> relPos)
            ( mkT5RelPos @inputSeqDim @T5RelPosEncBucketDim
            )
            inputSeqDim
            t5MaxDistance
        decoderRelPos =
          withoutDim @decoderInputSeqDim @(Int -> decoderRelPos)
            ( mkT5DecoderRelPos @decoderInputSeqDim @T5RelPosEncBucketDim
            )
            decoderInputSeqDim
            t5MaxDistance
     in forward seqToSeq (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask)
