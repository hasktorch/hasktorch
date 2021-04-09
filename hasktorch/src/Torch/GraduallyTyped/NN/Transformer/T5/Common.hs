{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RankNTypes #-}
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
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.RewriteRules.LayoutDeviceRule
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.RewriteRules.LayoutDataTypeRule #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Common where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (MonadIO, MonadReader, ReaderT (runReaderT), ask, liftIO)
import Data.Coerce (Coercible, coerce)
import Data.Generics.Product.Positions (HasPosition, position)
import Data.Kind (Type)
import qualified Data.Map as Map
import Foreign.ForeignPtr (ForeignPtr)
import GHC.Float (double2Int)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat, Symbol, type (-), type (<=?))
import Lens.Family ((^.))
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice)
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Activation (Relu (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Linear (Linear (..))
import Torch.GraduallyTyped.NN.Normalization (LayerNorm (..))
import Torch.GraduallyTyped.NN.Sparse (Embedding (..))
import Torch.GraduallyTyped.NN.Transformer.Block (TransformerBlock (TransformerBlock))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttention (..), GCrossAttention (..))
import Torch.GraduallyTyped.NN.Transformer.Decoder (GTransformerDecoder (..), TransformerDecoder (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (TransformerDecoderBlock (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack, TransformerDecoderStack (..))
import Torch.GraduallyTyped.NN.Transformer.Encoder (GTransformerEncoder (..), TransformerEncoder (..))
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (GTransformerFeedForwardNetwork (..), TransformerFeedForwardNetwork (..))
import Torch.GraduallyTyped.NN.Transformer.MultiHeadAttention (GMultiHeadAttention (..), MultiHeadAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (GSelfAttention (..), SelfAttention (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer (..), SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..), SequenceToSequenceTransformerWithLMHead (..), lookupSequenceToSequenceTransformer, lookupSequenceToSequenceTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack, TransformerStack (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, ShiftRight, TensorDict, TransformerStyle (T5), mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.RewriteRules ()
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, ReplaceDimF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape, Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), full, ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, cat, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (logicalOr)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape, dataType, device, layout, shape)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList
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
type T5RelPosEncBucketDim = 'Dim ('Name "*") ('Size 32)

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

-- | T5 begin-of-sentence token id.
t5BOSTokenId :: Int
t5BOSTokenId = t5PadTokenId

-- | T5 end-of-sentence token id.
-- 'eos_token_id = 1'
t5EOSTokenId :: Int
t5EOSTokenId = 1

-- | T5 attention mask bias
t5AttentionMaskBias :: Double
t5AttentionMaskBias = -10000

-- | T5 model.
newtype
  T5Model
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  T5Model ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    T5ModelSeqToSeqF T5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    T5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

-- | T5 model with language modelling head.
newtype
  T5ModelWithLMHead
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  T5ModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    T5ModelSeqToSeqF T5ModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    T5ModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

type family
  T5ModelSeqToSeqF
    ( t5Model ::
        Nat ->
        Device (DeviceType Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Dim (Name Symbol) (Size Nat) ->
        Type
    )
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  T5ModelSeqToSeqF T5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer
      numLayers
      numLayers
      'T5
      device
      T5DataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      T5RelPosEncBucketDim
      vocabDim
      T5DropoutP
  T5ModelSeqToSeqF T5ModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformerWithLMHead
      numLayers
      numLayers
      'T5
      device
      T5DataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      T5RelPosEncBucketDim
      vocabDim
      T5DropoutP

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'T5 ('Device 'CPU) T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5DropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'T5 ('Device 'CPU) T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim T5DropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (T5Model numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (T5Model numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (T5Model numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        T5Model <$> lookupSequenceToSequenceTransformer t5DropoutP t5Eps ""

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'T5 ('Device 'CPU) T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5DropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'T5 ('Device 'CPU) T5DataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim T5DropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (T5ModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (T5ModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (T5ModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        T5ModelWithLMHead <$> lookupSequenceToSequenceTransformerWithLMHead t5DropoutP t5Eps ""

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
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  WithDimF batchDim (WithDimF seqDim ([[Int]] -> m output))
mkT5Input = mkTransformerInput @batchDim @seqDim @m t5PadTokenId

mkT5PaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkT5PaddingMask = mkTransformerPaddingMask t5PadTokenId

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
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
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
                  pure (UnsafeTensor @'WithoutGradient t)
                    >>= checkedLayout @('Layout 'Dense)
                    >>= checkedDevice @('Device 'CPU)
                    >>= checkedDataType @('DataType 'Int64)
                    >>= checkedShape @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])

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
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
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
                  pure (UnsafeTensor @'WithoutGradient t)
                    >>= checkedLayout @('Layout 'Dense)
                    >>= checkedDevice @('Device 'CPU)
                    >>= checkedDataType @('DataType 'Int64)
                    >>= checkedShape @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])

data T5Input input decoderInput where
  T5Input ::
    forall input decoderInput.
    { t5Input :: input,
      t5DecoderInput :: decoderInput
    } ->
    T5Input input decoderInput

deriving instance
  ( Show input,
    Show decoderInput
  ) =>
  Show (T5Input input decoderInput)

data T5Output decoderOutput encoderOutput inputPaddingMask where
  T5Output ::
    forall decoderOutput encoderOutput inputPaddingMask.
    { t5DecoderOutput :: decoderOutput,
      t5EncoderOutput :: encoderOutput,
      t5InputPaddingMask :: inputPaddingMask
    } ->
    T5Output decoderOutput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderOutput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (T5Output decoderOutput encoderOutput inputPaddingMask)

data T5GenerationInput decoderInput encoderOutput inputPaddingMask where
  T5GenerationInput ::
    forall decoderInput encoderOutput inputPaddingMask.
    { t5GenerationDecoderInput :: decoderInput,
      t5GenerationEncoderOutput :: encoderOutput,
      t5GenerationInputPaddingMask :: inputPaddingMask
    } ->
    T5GenerationInput decoderInput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (T5GenerationInput decoderInput encoderOutput inputPaddingMask)

-- | 'HasForward' instance for T5 models.
--
-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
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
    inputPaddingMaskDataType ~ Seq (inputDataType <+> 'DataType 'Int64) ('DataType 'Bool),
    inputPaddingMaskShape ~ BroadcastShapesF inputShape ('Shape '[ 'Dim ('Name "*") ('Size 1)]),
    inputPaddingMaskSeqDim ~ (inputPaddingMaskShape ! 1),
    relPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), inputSeqDim, inputSeqDim]),
    WithDimC inputSeqDim (Int -> relPos),
    decoderInput
      ~ Tensor
          decoderInputRequiresGradient
          decoderInputLayout
          decoderInputDevice
          decoderInputDataType
          decoderInputShape,
    rightShiftedDecoderInput
      ~ Tensor
          rightShiftedDecoderInputRequiresGradient
          rightShiftedDecoderInputLayout
          rightShiftedDecoderInputDevice
          rightShiftedDecoderInputDataType
          rightShiftedDecoderInputShape,
    rightShiftedDecoderInputSeqDim ~ (rightShiftedDecoderInputShape ! 1),
    KnownDim rightShiftedDecoderInputSeqDim,
    KnownShape rightShiftedDecoderInputShape,
    decoderInputPaddingMask
      ~ Tensor
          'WithoutGradient
          (decoderInputLayout <+> 'Layout 'Dense)
          (decoderInputDevice <+> 'Device 'CPU)
          (Seq (decoderInputDataType <+> 'DataType 'Int64) ('DataType 'Bool))
          (BroadcastShapesF decoderInputShape ('Shape '[ 'Dim ('Name "*") ('Size 1)])),
    rightShiftedDecoderInputPaddingMask
      ~ Tensor
          rightShiftedDecoderInputPaddingMaskRequiresGradient
          rightShiftedDecoderInputPaddingMaskLayout
          rightShiftedDecoderInputPaddingMaskDevice
          rightShiftedDecoderInputPaddingMaskDataType
          rightShiftedDecoderInputPaddingMaskShape,
    rightShiftedDecoderInputPaddingMaskSeqDim ~ (rightShiftedDecoderInputPaddingMaskShape ! 1),
    decoderRelPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), rightShiftedDecoderInputSeqDim, rightShiftedDecoderInputSeqDim]),
    WithDimC rightShiftedDecoderInputSeqDim (Int -> decoderRelPos),
    MkTransformerAttentionMaskC T5DType T5DataType inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC T5DType T5DataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC T5DType T5DataType rightShiftedDecoderInputPaddingMaskRequiresGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (T5Input input decoderInput)
    generator
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward t5Model T5Input {..} =
    let inputPaddingMask = mkT5PaddingMask t5Input
        attentionMask =
          mkTransformerAttentionMask
            @T5DType
            @T5DataType
            @inputPaddingMaskRequiresGradient
            @inputPaddingMaskLayout
            @inputPaddingMaskDevice
            @inputPaddingMaskDataType
            @inputPaddingMaskShape
            t5AttentionMaskBias
            inputPaddingMask
        [_, inputSeqDim] = shape t5Input
        relPos =
          withoutDim @inputSeqDim @(Int -> relPos)
            ( mkT5RelPos @inputSeqDim @T5RelPosEncBucketDim
            )
            inputSeqDim
            t5MaxDistance
     in runIxState $
          ireturn t5DecoderInput
            >>>= IxState . forward (initialize @(ShiftRight Int) t5BOSTokenId)
            >>>= ( \rightShiftedDecoderInput ->
                     let [_, rightShiftedDecoderInputSeqDim] = shape rightShiftedDecoderInput
                         decoderRelPos =
                           withoutDim @rightShiftedDecoderInputSeqDim @(Int -> decoderRelPos)
                             ( mkT5DecoderRelPos @rightShiftedDecoderInputSeqDim @T5RelPosEncBucketDim
                             )
                             rightShiftedDecoderInputSeqDim
                             t5MaxDistance
                         crossAttentionMask =
                           withoutDim @rightShiftedDecoderInputSeqDim @(inputPaddingMask -> crossAttentionMask)
                             ( mkTransformerCrossAttentionMask
                                 @T5DType
                                 @T5DataType
                                 @rightShiftedDecoderInputSeqDim
                                 @inputPaddingMaskRequiresGradient
                                 @inputPaddingMaskLayout
                                 @inputPaddingMaskDevice
                                 @inputPaddingMaskDataType
                                 @inputPaddingMaskShape
                                 t5AttentionMaskBias
                             )
                             rightShiftedDecoderInputSeqDim
                             inputPaddingMask
                      in ireturn (mkT5PaddingMask t5DecoderInput)
                           >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          mkTransformerDecoderAttentionMask
                                            @T5DType
                                            @T5DataType
                                            @rightShiftedDecoderInputPaddingMaskRequiresGradient
                                            @rightShiftedDecoderInputPaddingMaskLayout
                                            @rightShiftedDecoderInputPaddingMaskDevice
                                            @rightShiftedDecoderInputPaddingMaskDataType
                                            @rightShiftedDecoderInputPaddingMaskShape
                                            t5AttentionMaskBias
                                            rightShiftedDecoderInputPaddingMask
                                     in ireturn (SequenceToSequenceTransformerInput t5Input rightShiftedDecoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)
                                )
                           >>>= IxState . forward (coerce t5Model :: T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                           >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ T5Output decoderOutput encoderOutput inputPaddingMask
                                )
                 )

-- | 'HasForward' instance for T5 models.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
instance
  ( decoderInput
      ~ Tensor
          decoderInputRequiresGradient
          decoderInputLayout
          decoderInputDevice
          decoderInputDataType
          decoderInputShape,
    rightShiftedDecoderInput
      ~ Tensor
          rightShiftedDecoderInputRequiresGradient
          rightShiftedDecoderInputLayout
          rightShiftedDecoderInputDevice
          rightShiftedDecoderInputDataType
          rightShiftedDecoderInputShape,
    rightShiftedDecoderInputSeqDim ~ (rightShiftedDecoderInputShape ! 1),
    KnownDim rightShiftedDecoderInputSeqDim,
    KnownShape rightShiftedDecoderInputShape,
    inputPaddingMask
      ~ Tensor
          inputPaddingMaskRequiresGradient
          inputPaddingMaskLayout
          inputPaddingMaskDevice
          inputPaddingMaskDataType
          inputPaddingMaskShape,
    KnownLayout inputPaddingMaskLayout,
    KnownDevice inputPaddingMaskDevice,
    KnownDataType inputPaddingMaskDataType,
    KnownShape inputPaddingMaskShape,
    decoderInputPaddingMask
      ~ Tensor
          'WithoutGradient
          (decoderInputLayout <+> 'Layout 'Dense)
          (decoderInputDevice <+> 'Device 'CPU)
          (Seq (decoderInputDataType <+> 'DataType 'Int64) ('DataType 'Bool))
          (BroadcastShapesF decoderInputShape ('Shape '[ 'Dim ('Name "*") ('Size 1)])),
    rightShiftedDecoderInputPaddingMask
      ~ Tensor
          rightShiftedDecoderInputPaddingMaskRequiresGradient
          rightShiftedDecoderInputPaddingMaskLayout
          rightShiftedDecoderInputPaddingMaskDevice
          rightShiftedDecoderInputPaddingMaskDataType
          rightShiftedDecoderInputPaddingMaskShape,
    rightShiftedDecoderInputPaddingMaskSeqDim ~ (rightShiftedDecoderInputPaddingMaskShape ! 1),
    decoderRelPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), rightShiftedDecoderInputSeqDim, rightShiftedDecoderInputSeqDim]),
    WithDimC rightShiftedDecoderInputSeqDim (Int -> decoderRelPos),
    MkTransformerCrossAttentionMaskC T5DType T5DataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC T5DType T5DataType rightShiftedDecoderInputPaddingMaskRequiresGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward t5Model T5GenerationInput {..} =
    runIxState $
      ireturn t5GenerationDecoderInput
        >>>= IxState . forward (initialize @(ShiftRight Int) t5BOSTokenId)
        >>>= ( \rightShiftedDecoderInput ->
                 let [_, rightShiftedDecoderInputSeqDim] = shape rightShiftedDecoderInput
                     decoderRelPos =
                       withoutDim @rightShiftedDecoderInputSeqDim @(Int -> decoderRelPos)
                         ( mkT5DecoderRelPos @rightShiftedDecoderInputSeqDim @T5RelPosEncBucketDim
                         )
                         rightShiftedDecoderInputSeqDim
                         t5MaxDistance
                     crossAttentionMask =
                       withoutDim @rightShiftedDecoderInputSeqDim @(inputPaddingMask -> crossAttentionMask)
                         ( mkTransformerCrossAttentionMask
                             @T5DType
                             @T5DataType
                             @rightShiftedDecoderInputSeqDim
                             @inputPaddingMaskRequiresGradient
                             @inputPaddingMaskLayout
                             @inputPaddingMaskDevice
                             @inputPaddingMaskDataType
                             @inputPaddingMaskShape
                             t5AttentionMaskBias
                         )
                         rightShiftedDecoderInputSeqDim
                         t5GenerationInputPaddingMask
                  in ireturn (mkT5PaddingMask t5GenerationDecoderInput)
                       >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      mkTransformerDecoderAttentionMask
                                        @T5DType
                                        @T5DataType
                                        @rightShiftedDecoderInputPaddingMaskRequiresGradient
                                        @rightShiftedDecoderInputPaddingMaskLayout
                                        @rightShiftedDecoderInputPaddingMaskDevice
                                        @rightShiftedDecoderInputPaddingMaskDataType
                                        @rightShiftedDecoderInputPaddingMaskShape
                                        t5AttentionMaskBias
                                        rightShiftedDecoderInputPaddingMask
                                 in ireturn (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput t5GenerationEncoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)
                            )
                       >>>= IxState . forward (coerce t5Model :: T5ModelSeqToSeqF t5Model numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ T5Output decoderOutput encoderOutput t5GenerationInputPaddingMask
                            )
             )