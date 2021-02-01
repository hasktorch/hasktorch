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
{-# OPTIONS_GHC -v2
                -fplugin TypeLevel.Rewrite
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

module Torch.GraduallyTyped.NN.Transformer.T5 where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader, ReaderT (runReaderT), ask, liftIO)
import Data.Data (Proxy (Proxy))
import qualified Data.Map as Map
import Foreign.ForeignPtr (ForeignPtr)
import GHC.Float (double2Int)
import GHC.TypeLits (KnownNat, Nat, Symbol, natVal, type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..), WithLayoutC (..))
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
import Torch.GraduallyTyped.Random (mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.RewriteRules ()
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC, ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape)
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

type StateDict = Map.Map String (ForeignPtr ATen.Tensor)

data T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP where
  T5Config ::
    forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP.
    ( KnownNat numLayers,
      KnownDevice device,
      KnownDataType dataType,
      KnownDim headDim,
      KnownDim headEmbedDim,
      KnownDim embedDim,
      KnownDim inputEmbedDim,
      KnownDim ffnDim,
      KnownDim relPosEncBucketDim,
      KnownDim vocabDim,
      Scalar dropoutP
    ) =>
    { debug :: Bool,
      dropoutP :: dropoutP,
      eps :: Double,
      maxDistance :: Int,
      stateDict :: StateDict
    } ->
    T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP

-- | T5 dropout rate
-- 'dropout_rate = 0.1'
t5DropoutP :: Float
t5DropoutP = 0.1

-- | T5 layer-norm epsilon
-- 'layer_norm_epsilon = 1e-06'
t5Eps :: Double
t5Eps = 1e-6

-- | T5 maximum distance
t5MaxDistance :: Int
t5MaxDistance = 128

t5ConfigFromPretrained ::
  (KnownNat numLayers, KnownDim headDim, KnownDim headEmbedDim, KnownDim embedDim, KnownDim inputEmbedDim, KnownDim ffnDim, KnownDim relPosEncBucketDim, KnownDim vocabDim) =>
  FilePath ->
  Bool ->
  IO (T5Config numLayers ( 'Device 'CPU) ( 'DataType 'Float) headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim Float)
t5ConfigFromPretrained filePath debug = do
  iValue <- Torch.Serialize.pickleLoad filePath
  stateDict <- case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a state dictionary."
  pure $ T5Config debug t5DropoutP t5Eps t5MaxDistance stateDict
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."

type T5AttentionMask device dataType batchSize inputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

mkT5AttentionMask ::
  forall device dataType inputSeqSize.
  WithCreateC (T5AttentionMask device dataType 1 inputSeqSize) 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)]) =>
  WithDeviceF device (WithDataTypeF dataType (T5AttentionMask device dataType 1 inputSeqSize))
mkT5AttentionMask =
  zeros
    @ 'WithoutGradient
    @( 'Layout 'Dense)
    @device
    @dataType
    @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size inputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

type T5DecoderAttentionMask device dataType batchSize decoderInputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])

mkT5DecoderAttentionMask ::
  forall device dataType decoderInputSeqSize.
  _ =>
  WithDeviceF
    device
    ( WithDataTypeF
        dataType
        (T5DecoderAttentionMask device dataType 1 decoderInputSeqSize)
    )
mkT5DecoderAttentionMask =
  withDevice @device $
    \deviceType ->
      withDataType
      -- @dataType
      -- @(T5DecoderAttentionMask device dataType 1 decoderInputSeqSize)
      $
        \dType ->
          let causalMask =
                unsqueeze @( 'SelectDim ( 'ByIndex 0))
                  . bool
                  . triu 1
                  $ withoutDataType
                    @dataType
                    -- @( Tensor
                    --      'WithoutGradient
                    --      ( 'Layout 'Dense)
                    --      device
                    --      dataType
                    --      ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])
                    --  )
                    ( withoutDevice @device
                        ( ones
                            @ 'WithoutGradient
                            @( 'Layout 'Dense)
                            @device
                            @dataType
                            @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])
                        )
                        deviceType
                    )
                    dType
           in maskedFill causalMask (-10000 :: Double) $
                withoutDataType
                  @dataType
                  -- @(T5DecoderAttentionMask device dataType 1 decoderInputSeqSize)
                  ( withoutDevice @device
                      ( zeros
                          @ 'WithoutGradient
                          @( 'Layout 'Dense)
                          @device
                          @dataType
                          @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize)])
                      )
                      deviceType
                  )
                  dType

type T5CrossAttentionMask device dataType batchSize inputSeqSize decoderInputSeqSize =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

mkT5CrossAttentionMask ::
  forall device dataType inputSeqSize decoderInputSeqSize.
  WithCreateC (T5CrossAttentionMask device dataType 1 inputSeqSize decoderInputSeqSize) 'WithoutGradient ( 'Layout 'Dense) device dataType ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)]) =>
  WithDeviceF device (WithDataTypeF dataType (T5CrossAttentionMask device dataType 1 inputSeqSize decoderInputSeqSize))
mkT5CrossAttentionMask =
  zeros
    @ 'WithoutGradient
    @( 'Layout 'Dense)
    @device
    @dataType
    @( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize)])

type T5Input device dataType batchSize inputSeqSize inputEmbedDim =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size inputSeqSize), inputEmbedDim])

type T5DecoderInput device dataType batchSize decoderInputSeqSize inputEmbedDim =
  Tensor
    'WithoutGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), inputEmbedDim])

type T5DecoderOutput device dataType batchSize decoderInputSeqSize inputEmbedDim =
  Tensor
    'WithGradient
    ( 'Layout 'Dense)
    device
    dataType
    ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size decoderInputSeqSize), inputEmbedDim])

class
  HasLookupStack
    (n :: Nat)
    (isCons :: Bool)
    (numLayers :: Nat)
    device
    dataType
    headDim
    headEmbedDim
    embedDim
    inputEmbedDim
    ffnDim
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    dropoutP
    m
  where
  lookupEncoderStack' ::
    Integer ->
    m (TransformerStack n device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
  lookupDecoderStack' ::
    Integer ->
    m (TransformerDecoderStack n device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim dropoutP)

instance
  Applicative m =>
  HasLookupStack 0 'False numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  where
  lookupEncoderStack' _ = pure TransformerStackNil
  lookupDecoderStack' _ = pure TransformerDecoderStackNil

instance
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack (n - 1) (1 <=? (n - 1)) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  HasLookupStack n 'True numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  where
  lookupEncoderStack' n =
    TransformerStackCons
      <$> lookupEncoderBlock n
      <*> lookupEncoderStack' @(n - 1) @(1 <=? (n - 1)) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim @dropoutP (n + 1)
  lookupDecoderStack' n =
    TransformerDecoderStackCons
      <$> lookupDecoderBlock n
      <*> lookupDecoderStack' @(n - 1) @(1 <=? (n - 1)) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim @dropoutP (n + 1)

lookupEncoderStack ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m (TransformerStack numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
lookupEncoderStack = lookupEncoderStack' @numLayers @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim @dropoutP 0

lookupDecoderStack ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim dropoutP)
lookupDecoderStack = lookupDecoderStack' @numLayers @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @inputEmbedDim @ffnDim @relPosEncBucketDim @vocabDim @dropoutP 0

lookupEncoder ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m
    ( TransformerEncoder
        numLayers
        device
        dataType
        headDim
        headEmbedDim
        embedDim
        inputEmbedDim
        ffnDim
        relPosEncBucketDim
        dropoutP
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
        <*> pure (initialize @(Dropout dropoutP) dropoutP)
        <*> ( Embedding
                <$> lookupTensor "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            )

lookupDecoder ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
        <*> pure (initialize @(Dropout dropoutP) dropoutP)
        <*> ( Embedding
                <$> lookupTensor "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            )

lookupSequenceToSequenceTransformerWithoutLMHead ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m (SequenceToSequenceTransformer 'WithoutLMHead numLayers numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    HasLookupStack numLayers (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m
  ) =>
  m (SequenceToSequenceTransformer 'WithLMHead numLayers numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP requiresGradient layout shape m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m,
    KnownLayout layout,
    KnownShape shape
  ) =>
  String ->
  m (Tensor requiresGradient layout device dataType shape)
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
        >>= checkedDataType
        >>= checkedShape

lookupHeadDim ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
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
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m
  ) =>
  Integer ->
  m (TransformerBlock device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
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
                        <*> pure (initialize @(Dropout dropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("encoder.block." <> show n <> ".layer.0.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wi.weight"))
                <*> (LinearWithoutBias <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.DenseReluDense.wo.weight"))
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("encoder.block." <> show n <> ".layer.1.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
            )

lookupDecoderBlock ::
  forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadIO m,
    MonadFail m
  ) =>
  Integer ->
  m (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim dropoutP)
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
                        <*> pure (initialize @(Dropout dropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.0.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
            )
        <*> ( CrossAttention
                <$> ( MultiHeadAttention
                        <$> lookupHeadDim
                        <*> lookupHeadEmbedDim
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.q.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.k.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.v.weight"))
                        <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.EncDecAttention.o.weight"))
                        <*> pure (initialize @(Dropout dropoutP) dropoutP)
                    )
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.1.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
            )
        <*> ( TransformerFeedForwardNetwork
                <$> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wi.weight"))
                <*> (LinearWithoutBias <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.DenseReluDense.wo.weight"))
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
                <*> ( LayerNormWithoutBias
                        <$> lookupTensor ("decoder.block." <> show n <> ".layer.2.layer_norm.weight")
                        <*> pure t5Eps
                    )
                <*> pure (initialize @(Dropout dropoutP) dropoutP)
            )

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
  forall seqSize numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadFail m,
    KnownNat seqSize
  ) =>
  m
    ( Tensor
        'WithoutGradient
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Int64)
        ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size seqSize), 'Dim ( 'Name "*") ( 'Size seqSize)])
    )
mkT5RelPos = do
  let seqSize = fromIntegral . natVal $ Proxy @seqSize
  config <- ask
  case config of
    T5Config {..} -> do
      relPosEncBucketSize <- fromIntegral . dimSize <$> lookupRelPosEncBucketDim
      case Torch.Tensor.asTensor [mkT5RelPos' relPosEncBucketSize maxDistance seqSize seqSize] of
        Torch.Tensor.Unsafe t ->
          pure (UnsafeTensor t)
            >>= checkedLayout
            >>= checkedDevice
            >>= checkedDataType
            >>= checkedShape

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
  forall seqSize numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP m.
  ( MonadReader (T5Config numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) m,
    MonadFail m,
    KnownNat seqSize
  ) =>
  m
    ( Tensor
        'WithoutGradient
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Int64)
        ( 'Shape '[ 'Dim ( 'Name "*") ( 'Size 1), 'Dim ( 'Name "*") ( 'Size seqSize), 'Dim ( 'Name "*") ( 'Size seqSize)])
    )
mkT5DecoderRelPos = do
  let seqSize = fromIntegral . natVal $ Proxy @seqSize
  config <- ask
  case config of
    T5Config {..} -> do
      relPosEncBucketSize <- fromIntegral . dimSize <$> lookupRelPosEncBucketDim
      case Torch.Tensor.asTensor [mkT5DecoderRelPos' relPosEncBucketSize maxDistance seqSize seqSize] of
        Torch.Tensor.Unsafe t ->
          pure (UnsafeTensor t)
            >>= checkedLayout
            >>= checkedDevice
            >>= checkedDataType
            >>= checkedShape

-- | T5-Small number of layers.
-- 'num_layers = 6'
type T5SmallNumLayers = 6

-- | T5-Small number of attention heads.
-- 'n_heads = 8'
type T5SmallHeadDim = 'Dim ( 'Name "*") ( 'Size 8)

-- | T5-Small head embedding dimension.
-- 'd_kv = 64'
type T5SmallHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | T5-Small embedding dimension.
-- 'inner_dim = n_heads * d_kv = 512'
type T5SmallEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | T5-Small model dimension.
-- 'd_model = 512'
type T5SmallInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 512)

-- | T5-Small feed-forward network dimension.
-- 'd_ff = 2048'
type T5SmallFFNDim = 'Dim ( 'Name "*") ( 'Size 2048)

-- | T5-Small relative positional encoding bucket dimension.
-- 'relative_attention_num_buckets = 32'
type T5SmallRelPosEncBucketDim = 'Dim ( 'Name "*") ( 'Size 32)

-- | T5-Small vocabulary dimension.
-- 'vocab_size = 32128'
type T5SmallVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | T5-Small configuration data type.
-- Modelled after https://huggingface.co/t5-small/blob/main/config.json.
type T5SmallConfig device dataType =
  T5Config T5SmallNumLayers device dataType T5SmallHeadDim T5SmallHeadEmbedDim T5SmallEmbedDim T5SmallInputEmbedDim T5SmallFFNDim T5SmallRelPosEncBucketDim T5SmallVocabDim Float

-- | load a T5-Small configuration from a file
t5SmallConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5SmallConfig ( 'Device 'CPU) ( 'DataType 'Float))
t5SmallConfigFromPretrained = t5ConfigFromPretrained

-- | T5-Small data type.
-- Modelled after https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py.
data T5Small hasLMHead device dataType where
  T5Small ::
    forall hasLMHead device dataType.
    { t5SmallSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5SmallNumLayers
          T5SmallNumLayers
          device
          dataType
          T5SmallHeadDim
          T5SmallHeadEmbedDim
          T5SmallEmbedDim
          T5SmallInputEmbedDim
          T5SmallFFNDim
          T5SmallRelPosEncBucketDim
          T5SmallVocabDim
          Float
    } ->
    T5Small hasLMHead device dataType

instance HasInitialize (T5Small 'WithoutLMHead device dataType) where
  type
    InitializeF (T5Small 'WithoutLMHead device dataType) =
      T5SmallConfig device dataType -> IO (T5Small 'WithoutLMHead device dataType)
  initialize config =
    flip runReaderT config $
      T5Small <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5Small 'WithLMHead device dataType) where
  type
    InitializeF (T5Small 'WithLMHead device dataType) =
      T5SmallConfig device dataType -> IO (T5Small 'WithLMHead device dataType)
  initialize config =
    flip runReaderT config $
      T5Small <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5SmallNumLayers
        T5SmallNumLayers
        device
        dataType
        T5SmallHeadDim
        T5SmallHeadEmbedDim
        T5SmallEmbedDim
        T5SmallInputEmbedDim
        T5SmallFFNDim
        T5SmallRelPosEncBucketDim
        T5SmallVocabDim
        Float
    )
    inputs
    generator
    output
    generatorOutput =>
  HasForward
    (T5Small hasLMHead device dataType)
    inputs
    generator
    output
    generatorOutput
  where
  forward T5Small {..} = forward t5SmallSeqToSeq

-- | T5-Base number of layers.
-- 'num_layers = 12'
type T5BaseNumLayers = 12

-- | T5-Base number of attention heads.
-- 'n_heads = 12'
type T5BaseHeadDim = 'Dim ( 'Name "*") ( 'Size 12)

-- | T5-Base head embedding dimension.
-- 'd_kv = 64'
type T5BaseHeadEmbedDim = 'Dim ( 'Name "*") ( 'Size 64)

-- | T5-Base embedding dimension.
-- 'inner_dim = n_heads * d_kv = 768'
type T5BaseEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

-- | T5-Base model dimension.
-- 'd_model = 768'
type T5BaseInputEmbedDim = 'Dim ( 'Name "*") ( 'Size 768)

-- | T5-Base feed-forward network dimension.
-- 'd_ff = 3072'
type T5BaseFFNDim = 'Dim ( 'Name "*") ( 'Size 3072)

-- | T5-Base relative positional encoding bucket dimension.
-- 'relative_attention_num_buckets = 32'
type T5BaseRelPosEncBucketDim = 'Dim ( 'Name "*") ( 'Size 32)

-- | T5-Base vocabulary dimension.
-- 'vocab_size = 32128'
type T5BaseVocabDim = 'Dim ( 'Name "*") ( 'Size 32128)

-- | T5-Base configuration data type.
-- Modelled after https://huggingface.co/t5-base/blob/main/config.json.
type T5BaseConfig device dataType =
  T5Config T5BaseNumLayers device dataType T5BaseHeadDim T5BaseHeadEmbedDim T5BaseEmbedDim T5BaseInputEmbedDim T5BaseFFNDim T5BaseRelPosEncBucketDim T5BaseVocabDim Float

-- | load a T5-Base configuration from a file
t5BaseConfigFromPretrained ::
  -- | file path
  FilePath ->
  -- | whether or not debugging output will be printed to the terminal
  Bool ->
  -- | configuration value
  IO (T5BaseConfig ( 'Device 'CPU) ( 'DataType 'Float))
t5BaseConfigFromPretrained = t5ConfigFromPretrained

-- | T5-Base data type.
-- Modelled after https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py.
data T5Base hasLMHead device dataType where
  T5Base ::
    forall hasLMHead device dataType.
    { t5BaseSeqToSeq ::
        SequenceToSequenceTransformer
          hasLMHead
          T5BaseNumLayers
          T5BaseNumLayers
          device
          dataType
          T5BaseHeadDim
          T5BaseHeadEmbedDim
          T5BaseEmbedDim
          T5BaseInputEmbedDim
          T5BaseFFNDim
          T5BaseRelPosEncBucketDim
          T5BaseVocabDim
          Float
    } ->
    T5Base hasLMHead device dataType

instance HasInitialize (T5Base 'WithoutLMHead device dataType) where
  type
    InitializeF (T5Base 'WithoutLMHead device dataType) =
      T5BaseConfig device dataType -> IO (T5Base 'WithoutLMHead device dataType)
  initialize config =
    flip runReaderT config $
      T5Base <$> lookupSequenceToSequenceTransformerWithoutLMHead

instance HasInitialize (T5Base 'WithLMHead device dataType) where
  type
    InitializeF (T5Base 'WithLMHead device dataType) =
      T5BaseConfig device dataType -> IO (T5Base 'WithLMHead device dataType)
  initialize config =
    flip runReaderT config $
      T5Base <$> lookupSequenceToSequenceTransformerWithLMHead

instance
  HasForward
    ( SequenceToSequenceTransformer
        hasLMHead
        T5BaseNumLayers
        T5BaseNumLayers
        device
        dataType
        T5BaseHeadDim
        T5BaseHeadEmbedDim
        T5BaseEmbedDim
        T5BaseInputEmbedDim
        T5BaseFFNDim
        T5BaseRelPosEncBucketDim
        T5BaseVocabDim
        Float
    )
    inputs
    generator
    output
    generatorOutput =>
  HasForward
    (T5Base hasLMHead device dataType)
    inputs
    generator
    output
    generatorOutput
  where
  forward T5Base {..} = forward t5BaseSeqToSeq

mkT5Input ::
  forall batchSize seqSize m.
  (MonadFail m, KnownNat batchSize, KnownNat seqSize) =>
  [[Int]] ->
  m
    ( Tensor
        'WithoutGradient
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Int64)
        ( 'Shape
            '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size seqSize)]
        ),
      Tensor
        'WithoutGradient
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Bool)
        ( 'Shape
            '[ 'Dim ( 'Name "*") ( 'Size batchSize), 'Dim ( 'Name "*") ( 'Size seqSize)]
        )
    )
mkT5Input xs = do
  input <- case Torch.Tensor.asTensor xs of
    Torch.Tensor.Unsafe t ->
      pure (UnsafeTensor t)
        >>= checkedLayout
        >>= checkedDevice
        >>= checkedDataType
        >>= checkedShape
  let paddingMask = undefined
  pure (input, paddingMask)

testForwardT5Small :: IO ()
testForwardT5Small =
  do
    (input, inputPaddingMask) <- mkT5Input @1 @15 [[6536, 43, 118, 2008, 24, 293, 53, 3, 9, 1782, 19, 207, 21, 25, 1]]
    (decoderInput, decoderInputPaddingMask) <- mkT5Input @1 @4 [[6536, 504, 24, 1]]
    let attentionMask = mkT5AttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @15
        decoderAttentionMask = mkT5DecoderAttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @4
        crossAttentionMask = mkT5CrossAttentionMask @( 'Device 'CPU) @( 'DataType 'Float) @15 @4
    config <- t5SmallConfigFromPretrained "/Users/tscholak/Projects/thirdParty/hasktorch/hasktorch/src/Torch/GraduallyTyped/NN/Transformer/t5-small.pt" False
    relPos <- runReaderT (mkT5RelPos @15) config
    decoderRelPos <- runReaderT (mkT5DecoderRelPos @4) config
    model <- initialize @(T5Small 'WithLMHead _ _) config
    g <- mkGenerator @( 'Device CPU) 0
    let (output, _) = forward (t5SmallSeqToSeq model) (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask) g
    print output
    pure ()

foo = withLayout $ \layoutType -> withDevice $ \deviceType -> go layoutType deviceType
  where
    go layoutType deviceType = UnsafeTensor @ 'WithGradient @( 'Layout Dense) @( 'Device 'CPU) @( 'DataType 'Float) @( 'Shape '[]) undefined

bar = withDevice @( 'Device 'CPU) go
  where
    go deviceType = UnsafeTensor @ 'WithGradient @( 'Layout Dense) @_ @( 'DataType 'Float) @( 'Shape '[]) undefined
