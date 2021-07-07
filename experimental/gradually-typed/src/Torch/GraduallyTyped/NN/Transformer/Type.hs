{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections #-}
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

module Torch.GraduallyTyped.NN.Transformer.Type where

import Control.Monad.Catch (MonadThrow)
import Data.Singletons.Prelude.List (SList (SNil))
import Data.Singletons.TH (SingKind (fromSing), genSingletons)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..), HasStateDict (..))
import Torch.GraduallyTyped.Prelude (Seq, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, ReplaceDimF, sGetDim, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), Name (..), SBy (..), SDim (sDimSize), SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals, sFull, sOnes, sZeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, cat, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (logicalOr)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (SGetDataType (sDataType), SGetDevice (..), SGetDim, SGetLayout (..), SGetShape (..), Tensor (..), bool, sCheckedShape, toTensor)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList

data TransformerStyle = T5 | ByT5 | BART | MBART | Pegasus | BERT | RoBERTa | GPT2
  deriving (Show, Eq)

genSingletons [''TransformerStyle]

data TransformerHead = WithoutHead | WithLMHead | WithMLMHead

genSingletons [''TransformerHead]

padded :: Integral n => n -> a -> [a] -> [a]
padded n p xs =
  let n' = fromIntegral n
      diff = n' - length xs
   in take n' xs ++ replicate diff p

mkTransformerInput ::
  forall batchDim seqDim m output.
  ( MonadThrow m,
    SGetDim batchDim,
    SGetDim seqDim,
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          ('Layout 'Dense)
          ('Device 'CPU)
          ('DataType 'Int64)
          ('Shape '[batchDim, seqDim])
  ) =>
  -- | padding token id
  Int ->
  -- | batch dimension singleton
  SDim batchDim ->
  -- | sequence dimension singleton
  SDim seqDim ->
  -- | batch of input ids
  [[Int]] ->
  -- | input tensor
  m output
mkTransformerInput padTokenId batchDim seqDim xs = do
  let batchSize = (\(Dim _ size) -> forgetIsChecked size) $ fromSing batchDim
      seqSize = (\(Dim _ size) -> forgetIsChecked size) $ fromSing seqDim
      emptySeq = replicate (fromIntegral seqSize) padTokenId
      paddedXs = padded batchSize emptySeq (padded seqSize padTokenId <$> xs)
  toTensor @('Gradient 'WithoutGradient) @('Layout 'Dense) @('Device 'CPU) paddedXs
    >>= sCheckedShape (SShape $ batchDim :|: seqDim :|: SNil)

type MkPosC device shape seqDim seqName seqSize output =
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
          ('Shape '[ 'Dim ('Name "*") seqSize])
  )

mkPos ::
  forall m gradient layout device dataType shape seqDim seqName seqSize output.
  ( MonadThrow m,
    MkPosC device shape seqDim seqName seqSize output
  ) =>
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | positions of the input tokens
  m output
mkPos input = do
  let device = sDevice input
      shape = sShape input
  seqDim <- sGetDim (SSelectDim $ SByIndex @1) shape
  let seqSize = sDimSize seqDim
      pos =
        sArangeNaturals
          (SGradient SWithoutGradient)
          (SLayout SDense)
          device
          (SDataType SInt64)
          seqSize
  pure pos

type MkTransformerPaddingMaskC layout device dataType shape output =
  ( output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          (device <+> 'Device 'CPU)
          (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
          (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
  )

mkTransformerPaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  -- | padding token id
  Int ->
  -- | input tensor
  Tensor gradient layout device dataType shape ->
  -- | padding mask
  output
mkTransformerPaddingMask padTokenId input =
  let padToken =
        sFull
          (SGradient SWithoutGradient)
          (SLayout SDense)
          (SDevice SCPU)
          (SDataType SInt64)
          (SShape $ SName @"*" :&: SSize @1 :|: SNil)
          padTokenId
   in input ==. padToken

type MkTransformerAttentionMaskC transformerDataType gradient layout device dataType shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          (Seq (gradient <+> 'Gradient 'WithoutGradient) ('Gradient 'WithoutGradient))
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) transformerDataType)
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          )
  )

mkTransformerAttentionMask ::
  forall m transformerDataType gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerAttentionMaskC transformerDataType gradient layout device dataType shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | encoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerAttentionMask transformerDataType attentionMaskBias paddingMask = do
  let pmLayout = sLayout paddingMask
      pmDevice = sDevice paddingMask
      pmShape = sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let emptyMask = sZeros (SGradient SWithoutGradient) pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
  pure $ maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

type MkTransformerDecoderAttentionMaskC transformerDataType layout device shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          ('Gradient 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          transformerDataType
          ( BroadcastShapesF
              ( BroadcastShapesF
                  ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
                  (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              )
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          )
  )

mkTransformerDecoderAttentionMask ::
  forall m transformerDataType gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerDecoderAttentionMaskC transformerDataType layout device shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | decoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerDecoderAttentionMask transformerDataType attentionMaskBias paddingMask = do
  let pmLayout = sLayout paddingMask
      pmDevice = sDevice paddingMask
      pmShape = sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let causalMask =
        unsqueeze @('SelectDim ('ByIndex 0))
          . bool
          . triu 1
          $ sOnes (SGradient SWithoutGradient) pmLayout pmDevice transformerDataType (SShape $ pmSeqDim :|: pmSeqDim :|: SNil)
      emptyMask = sZeros (SGradient SWithoutGradient) pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
      booleanMask = causalMask `logicalOr` unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
  pure $
    maskedFill
      booleanMask
      attentionMaskBias
      emptyMask

type MkTransformerCrossAttentionMaskC transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output =
  ( SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    SGetShape decoderInputShape,
    decoderInputSeqDim ~ (decoderInputShape ! 1),
    output
      ~ Tensor
          (Seq (gradient <+> 'Gradient 'WithoutGradient) ('Gradient 'WithoutGradient))
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) transformerDataType)
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), decoderInputSeqDim, seqDim])
          )
  )

mkTransformerCrossAttentionMask ::
  forall m transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output.
  ( MonadThrow m,
    MkTransformerCrossAttentionMaskC transformerDataType decoderInputShape decoderInputSeqDim gradient layout device dataType shape seqDim output
  ) =>
  -- | data type singleton of the transformer
  SDataType transformerDataType ->
  -- | decoder input shape
  SShape decoderInputShape ->
  -- | attention mask bias (typically a large negative number)
  Double ->
  -- | encoder padding mask
  Tensor gradient layout device dataType shape ->
  m output
mkTransformerCrossAttentionMask transformerDataType decoderInputShape attentionMaskBias paddingMask = do
  decoderInputSeqDim <- sGetDim (SSelectDim $ SByIndex @1) decoderInputShape
  let pmLayout = sLayout paddingMask
      pmDevice = sDevice paddingMask
      pmShape = sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let emptyMask = sZeros (SGradient SWithoutGradient) pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderInputSeqDim :|: pmSeqDim :|: SNil)
  pure $ maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

data ShiftRight fillValue where
  ShiftRight :: forall fillValue. fillValue -> ShiftRight fillValue

instance HasInitialize (ShiftRight fillValue) fillValue generator generator where
  initialize fillValue = (ShiftRight fillValue,)

instance HasStateDict (ShiftRight fillValue) fillValue where
  fromStateDict fillValue _ = pure $ ShiftRight fillValue
  toStateDict _ _ = pure ()

instance
  ( input
      ~ Tensor
          inputGradient
          inputLayout
          inputDevice
          inputDataType
          inputShape,
    SGetLayout inputLayout,
    SGetDevice inputDevice,
    SGetDataType inputDataType,
    SGetShape inputShape,
    inputBatchDim ~ (inputShape ! 0),
    inputSeqDim ~ (inputShape ! 1),
    Scalar fillValue,
    rightShiftedInput
      ~ Tensor
          (inputGradient <|> 'Gradient 'WithoutGradient)
          inputLayout
          inputDevice
          inputDataType
          ( ReplaceDimF
              ('SelectDim ('ByIndex 1))
              (inputShape <+> 'Shape '[inputBatchDim, inputSeqDim])
              (AddDimF inputSeqDim ('Dim ('Name "*") ('Size 1)))
          )
  ) =>
  HasForward (ShiftRight fillValue) input generator rightShiftedInput generator
  where
  forward (ShiftRight fillValue) input g = unsafePerformIO $ do
    let inputLayout = sLayout input
        inputDevice = sDevice input
        inputDataType = sDataType input
        inputShape = sShape input
    inputBatchDim <- sGetDim (SSelectDim $ SByIndex @0) inputShape
    let filler = sFull (SGradient SWithoutGradient) inputLayout inputDevice inputDataType (SShape $ inputBatchDim :|: SName @"*" :&: SSize @1 :|: SNil) fillValue
    pure (cat @('SelectDim ('ByIndex 1)) (filler :. input :. HNil), g)
