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
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeFamilyDependencies #-}
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

module Torch.GraduallyTyped.NN.Transformer.T5.Common where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..), SingKind (fromSing))
import Data.Singletons.Prelude.List (SList (..))
import GHC.Float (double2Int)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer (..), SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, MkTransformerPaddingMaskC, ShiftRight (..), TransformerHead (..), TransformerStyle (ByT5, T5), mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Prelude (forgetIsChecked)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (sGetDim, type (!))
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (SGetDevice (sDevice), SGetShape, Tensor (..), UncheckedTensor, checkedDataType, checkedGradient, checkedLayout, sCheckedDevice, sCheckedShape, sShape)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

-- | T5 dType.
type T5DType = 'Float

-- | T5 dType singleton.
t5DType :: SDType T5DType
t5DType = sing @T5DType

-- | T5 data type.
type T5DataType = 'DataType T5DType

-- | T5 data type singleton.
t5DataType :: SDataType T5DataType
t5DataType = sing @T5DataType

-- | T5 dropout probability type.
type T5DropoutP = Float

-- | T5 dropout rate.
-- 'dropout_rate = 0.1'
t5DropoutP :: T5DropoutP
t5DropoutP = 0.1

-- | T5 relative positional encoding bucket dimension.
-- 'relative_attention_num_buckets = 32'
type T5RelPosEncBucketDim = 'Dim ('Name "*") ('Size 32)

-- | T5 relative positional encoding bucket dimension singleton.
t5RelPosEncBucketDim :: SDim T5RelPosEncBucketDim
t5RelPosEncBucketDim = sing @T5RelPosEncBucketDim

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

data
  GT5Model
    (t5Model :: Type)
  where
  GT5Model ::
    forall t5Model.
    { t5Model :: t5Model,
      t5ShiftRightDecoderInput :: ShiftRight Int,
      t5ShiftRightPaddingMask :: ShiftRight Int
    } ->
    GT5Model t5Model

-- | T5 model.
data
  T5Model
    (style :: TransformerStyle)
    (transformerHead :: TransformerHead)
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  T5Model ::
    forall style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    GT5Model
      (T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) ->
    T5Model style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

type T5ModelF :: TransformerStyle -> TransformerHead -> Nat -> Nat -> Gradient RequiresGradient -> Device (DeviceType Nat) -> Dim (Name Symbol) (Size Nat) -> Dim (Name Symbol) (Size Nat) -> Dim (Name Symbol) (Size Nat) -> Dim (Name Symbol) (Size Nat) -> Dim (Name Symbol) (Size Nat) -> Dim (Name Symbol) (Size Nat) -> Type
type family
  T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    r | r -> style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  where
  T5ModelF 'T5 transformerHead numLayers numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer 'T5 transformerHead numLayers numLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim vocabDim T5DropoutP
  T5ModelF 'ByT5 transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer 'ByT5 transformerHead numEncoderLayers numDecoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim vocabDim T5DropoutP

instance
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim,
    HasStateDict
      (T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SGradient gradient, SDevice device, SDataType T5DataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim T5RelPosEncBucketDim, SDim vocabDim, T5DropoutP, Double)
  ) =>
  HasStateDict
    (T5Model style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (SGradient gradient, SDevice device)
  where
  fromStateDict (gradient, device) k =
    let headDim = sing @headDim
        headEmbedDim = sing @headEmbedDim
        embedDim = sing @embedDim
        inputEmbedDim = sing @inputEmbedDim
        ffnDim = sing @ffnDim
        vocabDim = sing @vocabDim
     in T5Model
          <$> ( GT5Model
                  <$> fromStateDict (gradient, device, t5DataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, t5RelPosEncBucketDim, vocabDim, t5DropoutP, t5Eps) k
                  <*> fromStateDict t5BOSTokenId k
                  <*> fromStateDict 0 k
              )
  toStateDict k (T5Model GT5Model {..}) = do
    toStateDict k t5Model
    toStateDict k t5ShiftRightDecoderInput
    toStateDict k t5ShiftRightPaddingMask

mkT5Input ::
  forall batchDim seqDim m output.
  ( MonadThrow m,
    KnownDim batchDim,
    KnownDim seqDim,
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  SDim batchDim ->
  SDim seqDim ->
  [[Int]] ->
  m output
mkT5Input = mkTransformerInput t5PadTokenId

mkT5PaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  Tensor gradient layout device dataType shape ->
  output
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

type MkT5RelPosC device shape seqDim seqName seqSize output =
  ( SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    seqDim ~ 'Dim seqName seqSize,
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          device
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") seqSize, 'Dim ('Name "*") seqSize])
  )

mkT5RelPos ::
  forall m gradient layout device dataType shape seqDim seqName seqSize output.
  ( MonadThrow m,
    MkT5RelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkT5RelPos input = do
  let device = sDevice input
      shape = sShape input
  seqDim <- sGetDim (SSelectDim $ SByIndex @1) shape
  let seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
      relPosEncBucketSize = fromInteger . forgetIsChecked . fromSing $ sDimSize t5RelPosEncBucketDim
  case Torch.Tensor.asTensor [mkT5RelPos' relPosEncBucketSize t5MaxDistance seqSize seqSize] of
    Torch.Tensor.Unsafe t ->
      pure (UnsafeTensor t :: UncheckedTensor)
        >>= checkedGradient
        >>= checkedLayout
        >>= sCheckedDevice device
        >>= checkedDataType
        >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)

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
  forall m gradient layout device dataType shape seqDim seqName seqSize output.
  ( MonadThrow m,
    MkT5RelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | decoder input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkT5DecoderRelPos input = do
  let device = sDevice input
      shape = sShape input
  seqDim <- sGetDim (SSelectDim $ SByIndex @1) shape
  let seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
      relPosEncBucketSize = fromInteger . forgetIsChecked . fromSing $ sDimSize t5RelPosEncBucketDim
  case Torch.Tensor.asTensor [mkT5DecoderRelPos' relPosEncBucketSize t5MaxDistance seqSize seqSize] of
    Torch.Tensor.Unsafe t ->
      pure (UnsafeTensor t :: UncheckedTensor)
        >>= checkedGradient
        >>= checkedLayout
        >>= sCheckedDevice device
        >>= checkedDataType
        >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)

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

-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
instance
  ( input ~ Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
    MkT5RelPosC inputDevice inputShape inputSeqDim inputSeqName inputSeqSize pos,
    MkTransformerPaddingMaskC inputLayout inputDevice inputDataType inputShape inputPaddingMask,
    inputPaddingMask ~ Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape,
    decoderInput ~ Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
    rightShiftedDecoderInput ~ Tensor rightShiftedDecoderInputGradient rightShiftedDecoderInputLayout rightShiftedDecoderInputDevice rightShiftedDecoderInputDataType rightShiftedDecoderInputShape,
    MkT5RelPosC rightShiftedDecoderInputDevice rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize decoderPos,
    MkTransformerPaddingMaskC decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape decoderInputPaddingMask,
    rightShiftedDecoderInputPaddingMask ~ Tensor rightShiftedDecoderInputPaddingMaskGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape,
    MkTransformerAttentionMaskC T5DataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC T5DataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC T5DataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (T5Model style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (T5Input input decoderInput)
    generator
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (T5Model GT5Model {..}) T5Input {..} =
    let inputPaddingMask = mkT5PaddingMask t5Input
        attentionMask = ilift $ mkTransformerAttentionMask t5DataType t5AttentionMaskBias inputPaddingMask
        relPos = ilift $ mkT5RelPos t5Input
     in runIxStateT $
          ireturn t5DecoderInput
            >>>= IxStateT . forward t5ShiftRightDecoderInput
            >>>= ( \rightShiftedDecoderInput ->
                     let decoderRelPos =
                           ilift $ mkT5DecoderRelPos rightShiftedDecoderInput
                         crossAttentionMask =
                           ilift $
                             mkTransformerCrossAttentionMask
                               t5DataType
                               (sShape rightShiftedDecoderInput)
                               t5AttentionMaskBias
                               inputPaddingMask
                      in ireturn (mkT5PaddingMask t5DecoderInput)
                           >>>= IxStateT . forward t5ShiftRightPaddingMask
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          ilift $
                                            mkTransformerDecoderAttentionMask
                                              t5DataType
                                              t5AttentionMaskBias
                                              rightShiftedDecoderInputPaddingMask
                                     in SequenceToSequenceTransformerInput
                                          <<$>> ireturn t5Input
                                          <<*>> ireturn rightShiftedDecoderInput
                                          <<*>> relPos
                                          <<*>> decoderRelPos
                                          <<*>> attentionMask
                                          <<*>> decoderAttentionMask
                                          <<*>> crossAttentionMask
                                )
                           >>>= IxStateT . forward t5Model
                           >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ T5Output decoderOutput encoderOutput inputPaddingMask
                                )
                 )

testT5 = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      vocabDim = SName @"*" :&: SSize @32128
  g <- sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = sOnes (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
      attentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      decoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
      (t5Model, g') = initialize @(SequenceToSequenceTransformer 'T5 'WithLMHead 4 4 _ _ _ _ _ _ _ _ _ _ _) (gradient, device, t5DataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, t5RelPosEncBucketDim, vocabDim, t5DropoutP, t5Eps) g
  (t5Output, g'') <-
    let pos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
     in forward t5Model SequenceToSequenceTransformerInput {..} g'
  (t5Output', g''') <-
    let t5ShiftRightDecoderInput = ShiftRight t5BOSTokenId
        t5ShiftRightPaddingMask = ShiftRight 0
        model = T5Model (GT5Model {..})
        inputs = T5Input input decoderInput
     in forward model inputs g''
  pure ((t5Output, t5Output'), g''')

-- | 'HasForward' instance for T5 models.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
instance
  ( inputPaddingMask ~ Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape,
    decoderInput ~ Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
    rightShiftedDecoderInput ~ Tensor rightShiftedDecoderInputGradient rightShiftedDecoderInputLayout rightShiftedDecoderInputDevice rightShiftedDecoderInputDataType rightShiftedDecoderInputShape,
    MkT5RelPosC rightShiftedDecoderInputDevice rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize decoderPos,
    MkTransformerPaddingMaskC decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape decoderInputPaddingMask,
    rightShiftedDecoderInputPaddingMask ~ Tensor rightShiftedDecoderInputPaddingMaskGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape,
    MkTransformerAttentionMaskC T5DataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC T5DataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC T5DataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (T5Model style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (T5Model GT5Model {..}) T5GenerationInput {..} =
    runIxStateT $
      ireturn t5GenerationDecoderInput
        >>>= IxStateT . forward t5ShiftRightDecoderInput
        >>>= ( \rightShiftedDecoderInput ->
                 let decoderRelPos =
                       ilift $ mkT5DecoderRelPos rightShiftedDecoderInput
                     crossAttentionMask =
                       ilift $
                         mkTransformerCrossAttentionMask
                           t5DataType
                           (sShape rightShiftedDecoderInput)
                           t5AttentionMaskBias
                           t5GenerationInputPaddingMask
                  in ireturn (mkT5PaddingMask t5GenerationDecoderInput)
                       >>>= IxStateT . forward t5ShiftRightPaddingMask
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      ilift $
                                        mkTransformerDecoderAttentionMask
                                          t5DataType
                                          t5AttentionMaskBias
                                          rightShiftedDecoderInputPaddingMask
                                 in SequenceToSequenceTransformerGenerationInput
                                      <<$>> ireturn rightShiftedDecoderInput
                                      <<*>> ireturn t5GenerationEncoderOutput
                                      <<*>> decoderRelPos
                                      <<*>> decoderAttentionMask
                                      <<*>> crossAttentionMask
                            )
                       >>>= IxStateT . forward t5Model
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ T5Output decoderOutput encoderOutput t5GenerationInputPaddingMask
                            )
             )
