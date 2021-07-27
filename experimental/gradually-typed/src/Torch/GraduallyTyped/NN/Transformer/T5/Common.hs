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
import Data.Singletons.TypeLits (SNat (..))
import GHC.Float (double2Int)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (EDTDecoderF, EDTEncoderF, EDTHeadF, EDTSharedEmbeddingF, EncoderDecoderTransformerGenerationInput (..), EncoderDecoderTransformerInput (..), EncoderDecoderTransformerOutput (..), GEncoderDecoderTransformer (..), encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, MkTransformerPaddingMaskC, STransformerHead (SWithLMHead), STransformerStyle (SByT5, ST5), ShiftRight (..), TransformerHead (..), TransformerStyle (ByT5, T5), mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Prelude (Seq, forgetIsChecked)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (getDim, type (!))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SBy (..), SDim (..), SName (..), SSelectDim (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (SGetDevice (..), SGetDim, SGetShape, Tensor (..), TensorSpec (..), sCheckedShape, sShape, toTensor)
import Torch.GraduallyTyped.Unify (type (<+>))

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

-- | T5 dropout rate.
-- 'dropout_rate = 0.1'
t5DropoutP :: Double
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

-- | Generic T5 model data type.
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

type instance
  ModelSpec (GT5Model t5Model) =
    GT5Model (ModelSpec t5Model)

-- | Specifies the BART model.
type T5ModelF ::
  TransformerStyle ->
  TransformerHead ->
  Nat ->
  Nat ->
  Gradient RequiresGradient ->
  Device (DeviceType Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Dim (Name Symbol) (Size Nat) ->
  Type
type family
  T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  where
-- = r | r -> style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim

  T5ModelF 'T5 transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GT5Model
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'T5 numEncoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTDecoderF 'T5 numDecoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTSharedEmbeddingF 'T5 gradient device T5DataType inputEmbedDim vocabDim)
          (EDTHeadF 'T5 transformerHead gradient device T5DataType inputEmbedDim vocabDim)
      )
  T5ModelF 'ByT5 transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GT5Model
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'ByT5 numEncoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTDecoderF 'ByT5 numDecoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTSharedEmbeddingF 'ByT5 gradient device T5DataType inputEmbedDim vocabDim)
          (EDTHeadF 'ByT5 transformerHead gradient device T5DataType inputEmbedDim vocabDim)
      )

-- | Specifies the parameters of a T5 or ByT5 model.
--
-- - @transformerHead@: the head of the T5 or ByT5 model.
-- - @numLayers@: the number of layers in the T5 or ByT5 model.
-- - @gradient@: whether to compute the gradient of the T5 or ByT5 model.
-- - @device@: the computational device on which the T5 or ByT5 model parameters are to be allocated.
t5ModelSpec ::
  forall style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim
  ) =>
  STransformerStyle style ->
  STransformerHead transformerHead ->
  SNat numEncoderLayers ->
  SNat numDecoderLayers ->
  SGradient gradient ->
  SDevice device ->
  ModelSpec (T5ModelF style transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
t5ModelSpec style transformerHead numEncoderLayers numDecoderLayers gradient device =
  case style of
    ST5 ->
      GT5Model
        (modelSpec' ST5)
        (ShiftRight t5BOSTokenId)
        (ShiftRight 0)
    SByT5 ->
      GT5Model
        (modelSpec' SByT5)
        (ShiftRight t5BOSTokenId)
        (ShiftRight 0)
    _ -> undefined
  where
    modelSpec' :: _
    modelSpec' style' =
      encoderDecoderTransformerSpec
        style'
        transformerHead
        numEncoderLayers
        numDecoderLayers
        gradient
        device
        t5DataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        t5RelPosEncBucketDim
        (sing @vocabDim)
        t5DropoutP
        t5Eps

instance HasStateDict modelSpec => HasStateDict (GT5Model modelSpec) where
  fromStateDict (GT5Model modelSpec decoderInputShiftSpec paddingMaskShiftSpec) k =
    GT5Model
      <$> fromStateDict modelSpec k
      <*> fromStateDict decoderInputShiftSpec k
      <*> fromStateDict paddingMaskShiftSpec k
  toStateDict k GT5Model {..} = toStateDict k t5Model

mkT5Input ::
  forall batchDim seqDim m output.
  ( MonadThrow m,
    SGetDim batchDim,
    SGetDim seqDim,
    'Shape '[batchDim, seqDim]
      ~ Seq
          ( 'Shape
              '[ 'Dim ('Name "*") 'UncheckedSize,
                 'Dim ('Name "*") 'UncheckedSize
               ]
              <+> 'Shape '[batchDim, seqDim]
          )
          ('Shape '[batchDim, seqDim]),
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
    'Shape
      '[ 'Dim ('Name "*") ('Size 1),
         'Dim ('Name "*") seqSize,
         'Dim ('Name "*") seqSize
       ]
      ~ Seq
          ( '[ 'Dim ('Name "*") 'UncheckedSize,
               'Dim ('Name "*") 'UncheckedSize
             ]
              <+> '[ 'Dim ('Name "*") seqSize, 'Dim ('Name "*") seqSize]
          )
          ( 'Shape
              '[ 'Dim ('Name "*") ('Size 1),
                 'Dim ('Name "*") seqSize,
                 'Dim ('Name "*") seqSize
               ]
          ),
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
    SingI device,
    MkT5RelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkT5RelPos input =
  toTensor [mkT5RelPos' relPosEncBucketSize t5MaxDistance seqSize seqSize]
    >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)
  where
    seqDim = getDim (SSelectDim $ SByIndex @1) $ sShape input
    seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
    relPosEncBucketSize = fromInteger . forgetIsChecked . fromSing $ sDimSize t5RelPosEncBucketDim

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
    SingI device,
    MkT5RelPosC device shape seqDim seqName seqSize output
  ) =>
  -- | decoder input tensor
  Tensor gradient layout device dataType shape ->
  -- | relative positions of the input tokens
  m output
mkT5DecoderRelPos input =
  toTensor [mkT5DecoderRelPos' relPosEncBucketSize t5MaxDistance seqSize seqSize]
    >>= sCheckedShape (SShape $ SName @"*" :&: SSize @1 :|: SName @"*" :&: sDimSize seqDim :|: SName @"*" :&: sDimSize seqDim :|: SNil)
  where
    seqDim = getDim (SSelectDim $ SByIndex @1) $ sShape input
    seqSize = fromInteger . forgetIsChecked . fromSing $ sDimSize seqDim
    relPosEncBucketSize = fromInteger . forgetIsChecked . fromSing $ sDimSize t5RelPosEncBucketDim

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
    SingI inputDevice,
    SingI rightShiftedDecoderInputDevice,
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
    HasForward (ShiftRight Int) decoderInput generatorDevice rightShiftedDecoderInput generatorDevice,
    HasForward (ShiftRight Int) decoderInputPaddingMask generatorDevice rightShiftedDecoderInputPaddingMask generatorDevice,
    HasForward
      t5Model
      (EncoderDecoderTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GT5Model t5Model)
    (T5Input input decoderInput)
    generatorDevice
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutputDevice
  where
  forward GT5Model {..} T5Input {..} =
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
                                     in EncoderDecoderTransformerInput
                                          <<$>> ireturn t5Input
                                          <<*>> ireturn rightShiftedDecoderInput
                                          <<*>> relPos
                                          <<*>> decoderRelPos
                                          <<*>> attentionMask
                                          <<*>> decoderAttentionMask
                                          <<*>> crossAttentionMask
                                )
                           >>>= IxStateT . forward t5Model
                           >>>= ( \(EncoderDecoderTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ T5Output decoderOutput encoderOutput inputPaddingMask
                                )
                 )

-- | 'HasForward' instance for T5 models.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
instance
  ( inputPaddingMask ~ Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape,
    decoderInput ~ Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
    rightShiftedDecoderInput ~ Tensor rightShiftedDecoderInputGradient rightShiftedDecoderInputLayout rightShiftedDecoderInputDevice rightShiftedDecoderInputDataType rightShiftedDecoderInputShape,
    SingI rightShiftedDecoderInputDevice,
    MkT5RelPosC rightShiftedDecoderInputDevice rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize decoderPos,
    MkTransformerPaddingMaskC decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape decoderInputPaddingMask,
    rightShiftedDecoderInputPaddingMask ~ Tensor rightShiftedDecoderInputPaddingMaskGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape,
    MkTransformerAttentionMaskC T5DataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC T5DataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC T5DataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generatorDevice rightShiftedDecoderInput generatorDevice,
    HasForward (ShiftRight Int) decoderInputPaddingMask generatorDevice rightShiftedDecoderInputPaddingMask generatorDevice,
    HasForward
      t5Model
      (EncoderDecoderTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GT5Model t5Model)
    (T5GenerationInput decoderInput encoderOutput inputPaddingMask)
    generatorDevice
    (T5Output decoderOutput encoderOutput inputPaddingMask)
    generatorOutputDevice
  where
  forward GT5Model {..} T5GenerationInput {..} =
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
                                 in EncoderDecoderTransformerGenerationInput
                                      <<$>> ireturn rightShiftedDecoderInput
                                      <<*>> ireturn t5GenerationEncoderOutput
                                      <<*>> decoderRelPos
                                      <<*>> decoderAttentionMask
                                      <<*>> crossAttentionMask
                            )
                       >>>= IxStateT . forward t5Model
                       >>>= ( \(EncoderDecoderTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ T5Output decoderOutput encoderOutput t5GenerationInputPaddingMask
                            )
             )

testT5 :: IO _
testT5 = do
  let gradient = SGradient SWithGradient
      device = SDevice SCPU
      headDim = SName @"*" :&: SSize @8
      headEmbedDim = SName @"*" :&: SSize @64
      embedDim = SName @"*" :&: SSize @512
      inputEmbedDim = SName @"*" :&: SSize @512
      ffnDim = SName @"*" :&: SSize @2048
      vocabDim = SName @"*" :&: SSize @32128
  let g = sMkGenerator device 0
  let batchDim = SName @"*" :&: SSize @3
      seqDim = SName @"*" :&: SSize @13
      decoderSeqDim = SName @"*" :&: SSize @7
      sOnes' = (sOnes .) . TensorSpec (SGradient SWithoutGradient) (SLayout SDense) device
      input = sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
      attentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      decoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let spec = encoderDecoderTransformerSpec ST5 SWithLMHead (SNat @4) (SNat @4) gradient device t5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim t5RelPosEncBucketDim vocabDim t5DropoutP t5Eps
  (t5Model, g') <- initialize spec g
  (t5Output, g'') <-
    let pos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
     in forward t5Model EncoderDecoderTransformerInput {..} g'
  (t5Output', g''') <-
    let t5ShiftRightDecoderInput = ShiftRight t5BOSTokenId
        t5ShiftRightPaddingMask = ShiftRight 0
        model = GT5Model {..}
        inputs = T5Input input decoderInput
     in forward model inputs g''
  pure ((t5Output, t5Output'), g''')