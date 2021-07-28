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
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.T5.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.TypeLits (SNat (..))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (EDTDecoderF, EDTEncoderF, EDTHeadF, EDTSharedEmbeddingF, EncoderDecoderTransformerInput (..), GEncoderDecoderTransformer (..), GSimplifiedEncoderDecoderTransformer (..), SimplifiedEncoderDecoderTransformerInput (..), encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkRelPos (..), MkTransformerAttentionMask (..), MkTransformerCrossAttentionMask (..), MkTransformerDecoderAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead (SWithLMHead), STransformerStyle (SByT5, ST5), ShiftRight (..), TransformerHead (..), TransformerStyle (ByT5, T5), mkTransformerInput)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor (..), TensorSpec (..))
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

-- | Specifies a T5 or ByT5 model.
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
  T5ModelF 'T5 transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GSimplifiedEncoderDecoderTransformer
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'T5 numEncoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTDecoderF 'T5 numDecoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTSharedEmbeddingF 'T5 gradient device T5DataType inputEmbedDim vocabDim)
          (EDTHeadF 'T5 transformerHead gradient device T5DataType inputEmbedDim vocabDim)
      )
      (MkRelPos T5RelPosEncBucketDim)
      (MkRelPos T5RelPosEncBucketDim)
      MkTransformerPaddingMask
      (MkTransformerAttentionMask T5DataType)
      (MkTransformerCrossAttentionMask T5DataType)
      (MkTransformerDecoderAttentionMask T5DataType)
  T5ModelF 'ByT5 transformerHead numEncoderLayers numDecoderLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GSimplifiedEncoderDecoderTransformer
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'ByT5 numEncoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTDecoderF 'ByT5 numDecoderLayers gradient device T5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim T5RelPosEncBucketDim)
          (EDTSharedEmbeddingF 'ByT5 gradient device T5DataType inputEmbedDim vocabDim)
          (EDTHeadF 'ByT5 transformerHead gradient device T5DataType inputEmbedDim vocabDim)
      )
      (MkRelPos T5RelPosEncBucketDim)
      (MkRelPos T5RelPosEncBucketDim)
      MkTransformerPaddingMask
      (MkTransformerAttentionMask T5DataType)
      (MkTransformerCrossAttentionMask T5DataType)
      (MkTransformerDecoderAttentionMask T5DataType)

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
      GSimplifiedEncoderDecoderTransformer
        (modelSpec' ST5)
        (ShiftRight t5BOSTokenId)
        (ShiftRight 0)
        (MkRelPos t5RelPosEncBucketDim t5MaxDistance)
        (MkDecoderRelPos t5RelPosEncBucketDim t5MaxDistance)
        (MkTransformerPaddingMask t5PadTokenId)
        (MkTransformerAttentionMask t5DataType t5AttentionMaskBias)
        (MkTransformerCrossAttentionMask t5DataType t5AttentionMaskBias)
        (MkTransformerDecoderAttentionMask t5DataType t5AttentionMaskBias)
    SByT5 ->
      GSimplifiedEncoderDecoderTransformer
        (modelSpec' SByT5)
        (ShiftRight t5BOSTokenId)
        (ShiftRight 0)
        (MkRelPos t5RelPosEncBucketDim t5MaxDistance)
        (MkDecoderRelPos t5RelPosEncBucketDim t5MaxDistance)
        (MkTransformerPaddingMask t5PadTokenId)
        (MkTransformerAttentionMask t5DataType t5AttentionMaskBias)
        (MkTransformerCrossAttentionMask t5DataType t5AttentionMaskBias)
        (MkTransformerDecoderAttentionMask t5DataType t5AttentionMaskBias)
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

mkT5Input ::
  forall batchDim seqDim device m output.
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
          device
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  SDim batchDim ->
  SDim seqDim ->
  SDevice device ->
  [[Int]] ->
  m output
mkT5Input = mkTransformerInput t5PadTokenId

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
      edtInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: seqDim :|: SNil)
      edtAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      edtDecoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      edtDecoderAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      edtCrossAttentionMask = sOnes' t5DataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let spec = encoderDecoderTransformerSpec ST5 SWithLMHead (SNat @4) (SNat @4) gradient device t5DataType headDim headEmbedDim embedDim inputEmbedDim ffnDim t5RelPosEncBucketDim vocabDim t5DropoutP t5Eps
  (sedtModel, g') <- initialize spec g
  (t5Output, g'') <-
    let edtPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
        edtDecoderPos = sOnes' (SDataType SInt64) (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
     in forward sedtModel EncoderDecoderTransformerInput {..} g'
  (t5Output', g''') <-
    let sedtDecoderInputShift = ShiftRight t5BOSTokenId
        sedtPaddingMaskShift = ShiftRight 0
        sedtMkPos = MkRelPos t5RelPosEncBucketDim t5MaxDistance
        sedtMkDecoderPos = MkDecoderRelPos t5RelPosEncBucketDim t5MaxDistance
        sedtMkPaddingMask = MkTransformerPaddingMask t5PadTokenId
        sedtMkAttentionMask = MkTransformerAttentionMask t5DataType t5AttentionMaskBias
        sedtMkCrossAttentionMask = MkTransformerCrossAttentionMask t5DataType t5AttentionMaskBias
        sedtMkDecoderAttentionMask = MkTransformerDecoderAttentionMask t5DataType t5AttentionMaskBias
        model = GSimplifiedEncoderDecoderTransformer {..}
        inputs = SimplifiedEncoderDecoderTransformerInput edtInput edtDecoderInput
     in forward model inputs g''
  pure ((t5Output, t5Output'), g''')