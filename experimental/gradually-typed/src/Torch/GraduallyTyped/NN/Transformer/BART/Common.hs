{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Common where

import Control.Monad.Catch (MonadThrow)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.Prelude.List (SList (..))
import Data.Singletons.TypeLits (SNat (SNat))
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (initialize), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (EDTDecoderF, EDTEncoderF, EDTHeadF, EDTSharedEmbeddingF, EncoderDecoderTransformerInput (..), GEncoderDecoderTransformer, GSimplifiedEncoderDecoderTransformer (..), SimplifiedEncoderDecoderTransformerInput (..), encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkAbsPos (..), MkTransformerAttentionMask (..), MkTransformerCrossAttentionMask (..), MkTransformerDecoderAttentionMask (..), MkTransformerPaddingMask (..), STransformerHead (SWithLMHead), STransformerStyle (SBART), ShiftRight (..), TransformerHead (..), TransformerStyle (BART), mkTransformerInput)
import Torch.GraduallyTyped.Prelude (Seq, pattern (:|:))
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor, TensorSpec (..))
import Torch.GraduallyTyped.Unify (type (<+>))

-- | BART dType.
type BARTDType = 'Float

-- | BART dType singleton.
bartDType :: SDType BARTDType
bartDType = sing @BARTDType

-- | BART data type.
type BARTDataType = 'DataType BARTDType

-- | BART data type singleton.
bartDataType :: SDataType BARTDataType
bartDataType = sing @BARTDataType

-- | BART dropout rate.
-- 'dropout_rate = 0.1'
bartDropoutP :: Double
bartDropoutP = 0.1

-- | BART positional encoding dimension.
type BARTPosEncDim = 'Dim ('Name "*") ('Size 1026)

-- | BART positional encoding dimension singleton.
bartPosEncDim :: SDim BARTPosEncDim
bartPosEncDim = sing @BARTPosEncDim

-- | BART layer-norm epsilon.
bartEps :: Double
bartEps = 1e-5

-- | BART maximum number of position embeddings.
-- 'max_position_embeddings = 1024'
bartMaxPositionEmbeddings :: Int
bartMaxPositionEmbeddings = 1024

-- | BART padding token id.
-- 'pad_token_id = 1'
bartPadTokenId :: Int
bartPadTokenId = 1

-- | BART begin-of-sentence token id.
-- 'bos_token_id = 0'
bartBOSTokenId :: Int
bartBOSTokenId = 0

-- | BART end-of-sentence token id.
-- 'eos_token_id = 2'
bartEOSTokenId :: Int
bartEOSTokenId = 2

-- | BART attention mask bias
bartAttentionMaskBias :: Double
bartAttentionMaskBias = -10000

-- | Specifies the BART model.
type family
  BARTModelF
    (transformerHead :: TransformerHead)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat)) ::
    Type
  where
  BARTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GSimplifiedEncoderDecoderTransformer
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'BART numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim)
          (EDTDecoderF 'BART numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim)
          (EDTSharedEmbeddingF 'BART gradient device BARTDataType inputEmbedDim vocabDim)
          (EDTHeadF 'BART transformerHead gradient device BARTDataType inputEmbedDim vocabDim)
      )
      MkAbsPos
      MkAbsPos
      MkTransformerPaddingMask
      (MkTransformerAttentionMask BARTDataType)
      (MkTransformerCrossAttentionMask BARTDataType)
      (MkTransformerDecoderAttentionMask BARTDataType)

-- | Specifies the parameters of a BART model.
--
-- - @transformerHead@: the head of the BART model.
-- - @numLayers@: the number of layers in the BART model.
-- - @gradient@: whether to compute the gradient of the BART model.
-- - @device@: the computational device on which the BART model parameters are to be allocated.
bartModelSpec ::
  forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim
  ) =>
  STransformerHead transformerHead ->
  SNat numLayers ->
  SGradient gradient ->
  SDevice device ->
  ModelSpec (BARTModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
bartModelSpec transformerHead numLayers gradient device =
  GSimplifiedEncoderDecoderTransformer
    ( encoderDecoderTransformerSpec
        SBART
        transformerHead
        numLayers
        numLayers
        gradient
        device
        bartDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        bartPosEncDim
        (sing @vocabDim)
        bartDropoutP
        bartEps
    )
    (ShiftRight bartEOSTokenId)
    (ShiftRight 0)
    (MkAbsPosWithOffset 2)
    (MkAbsPosWithOffset 2)
    (MkTransformerPaddingMask bartPadTokenId)
    (MkTransformerAttentionMask bartDataType bartAttentionMaskBias)
    (MkTransformerCrossAttentionMask bartDataType bartAttentionMaskBias)
    (MkTransformerDecoderAttentionMask bartDataType bartAttentionMaskBias)

mkBARTInput ::
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
mkBARTInput = mkTransformerInput bartPadTokenId

testBart :: IO _
testBart = do
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
      edtAttentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      edtDecoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      edtDecoderAttentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      edtCrossAttentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
  let spec = encoderDecoderTransformerSpec SBART SWithLMHead (SNat @4) (SNat @4) gradient device bartDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim bartPosEncDim vocabDim bartDropoutP bartEps
  (sedtModel, g') <- initialize spec g
  (bartOutput, g'') <-
    let edtPos = sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
        edtDecoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
     in forward sedtModel EncoderDecoderTransformerInput {..} g'
  (bartOutput', g''') <-
    let sedtDecoderInputShift = ShiftRight bartEOSTokenId
        sedtPaddingMaskShift = ShiftRight 0
        sedtMkPos = MkAbsPosWithOffset 2
        sedtMkDecoderPos = MkAbsPosWithOffset 2
        sedtMkPaddingMask = MkTransformerPaddingMask bartPadTokenId
        sedtMkAttentionMask = MkTransformerAttentionMask bartDataType bartAttentionMaskBias
        sedtMkCrossAttentionMask = MkTransformerCrossAttentionMask bartDataType bartAttentionMaskBias
        sedtMkDecoderAttentionMask = MkTransformerDecoderAttentionMask bartDataType bartAttentionMaskBias
        model = GSimplifiedEncoderDecoderTransformer {..}
        inputs = SimplifiedEncoderDecoderTransformerInput edtInput edtDecoderInput
     in forward model inputs g''
  pure ((bartOutput, bartOutput'), g''')
