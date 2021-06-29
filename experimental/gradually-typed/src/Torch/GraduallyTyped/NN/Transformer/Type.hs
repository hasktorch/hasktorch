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

import Control.Monad.Reader (MonadIO, MonadReader, ask, liftIO)
import qualified Data.Map as Map
import Data.Singletons.Prelude.List (SList (SNil))
import Data.Singletons.TH (SingKind (fromSing), genSingletons)
import Foreign.ForeignPtr (ForeignPtr)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.Prelude (Seq, forgetIsChecked)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, ReplaceDimF, sGetDim, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SBy (..), SDim, SName (..), SSelectDim (..), SShape (..), SSize (..), SelectDim (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sFull, sOnes, sZeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, cat, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Comparison ((==.))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (logicalOr)
import Torch.GraduallyTyped.Tensor.Other (maskedFill, triu)
import Torch.GraduallyTyped.Tensor.Type (SGetDataType (sDataType), SGetDevice (..), SGetLayout (..), SGetShape (..), Tensor (..), bool, checkedDataType, checkedDevice, checkedLayout, checkedShape)
import Torch.GraduallyTyped.Unify (type (<+>), type (<|>))
import Torch.HList
import qualified Torch.Internal.Type as ATen (Tensor)
import qualified Torch.Script (IValue (..))
import qualified Torch.Serialize (pickleLoad)
import qualified Torch.Tensor (Tensor (Unsafe), asTensor)

data TransformerStyle = T5 | ByT5 | BART | MBART | BERT | RoBERTa | Pegasus | GPT2
  deriving (Show, Eq)

genSingletons [''TransformerStyle]

type TensorDict = Map.Map String (ForeignPtr ATen.Tensor)

tensorDictFromPretrained ::
  FilePath ->
  IO TensorDict
tensorDictFromPretrained filePath = do
  iValue <- Torch.Serialize.pickleLoad filePath
  case iValue of
    Torch.Script.IVGenericDict xs -> Map.fromList <$> go xs
    _ -> fail "iValue is not a tensor dictionary."
  where
    go [] = pure []
    go ((Torch.Script.IVString s, Torch.Script.IVTensor (Torch.Tensor.Unsafe t)) : xs) = ((s, t) :) <$> go xs
    go ((_, Torch.Script.IVTensor _) : _) = fail "iValue is not a string."
    go ((Torch.Script.IVString _, _) : _) = fail "iValue is not a tensor."
    go _ = fail "iValue is neither a string nor a tensor."

lookupTensor ::
  forall requiresGradient layout device dataType shape m.
  ( MonadReader TensorDict m,
    MonadIO m,
    MonadFail m,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape
  ) =>
  String ->
  m (Tensor requiresGradient layout device dataType shape)
lookupTensor s = do
  tensorDict <- ask
  liftIO
    ( maybe
        (fail $ "`" <> show s <> "` is not in the tensor dictionary.")
        (pure . UnsafeTensor)
        (Map.lookup s tensorDict)
    )
    >>= checkedLayout
    >>= checkedDevice
    >>= checkedDataType
    >>= checkedShape

padded :: Integral n => n -> a -> [a] -> [a]
padded n p xs =
  let n' = fromIntegral n
      diff = n' - length xs
   in take n' xs ++ replicate diff p

mkTransformerInput ::
  forall batchDim seqDim m output.
  ( MonadFail m,
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
  Int ->
  SDim batchDim ->
  SDim seqDim ->
  [[Int]] ->
  m output
mkTransformerInput padTokenId batchDim seqDim xs = do
  let batchSize = (\(Dim _ size) -> forgetIsChecked size) $ fromSing batchDim
      seqSize = (\(Dim _ size) -> forgetIsChecked size) $ fromSing seqDim
      emptySeq = replicate (fromIntegral seqSize) padTokenId
      paddedXs = padded batchSize emptySeq (padded seqSize padTokenId <$> xs)
  case Torch.Tensor.asTensor paddedXs of
    Torch.Tensor.Unsafe t ->
      pure (UnsafeTensor @'WithoutGradient t)
        >>= checkedLayout @('Layout 'Dense)
        >>= checkedDevice @('Device 'CPU)
        >>= checkedDataType @('DataType 'Int64)
        >>= checkedShape @('Shape '[batchDim, seqDim])

mkTransformerPaddingMask ::
  Int ->
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkTransformerPaddingMask padTokenId input =
  let padToken =
        sFull
          SWithoutGradient
          (SLayout SDense)
          (SDevice SCPU)
          (SDataType SInt64)
          (SShape $ SName @"*" :&: SSize @1 :|: SNil)
          padTokenId
   in input ==. padToken

type MkTransformerAttentionMaskC m transformerDataType requiresGradient layout device dataType shape seqDim output =
  ( MonadFail m,
    SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          (Seq (requiresGradient <+> 'WithoutGradient) 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) transformerDataType)
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          )
  )

mkTransformerAttentionMask ::
  forall m transformerDataType requiresGradient layout device dataType shape seqDim output.
  MkTransformerAttentionMaskC m transformerDataType requiresGradient layout device dataType shape seqDim output =>
  SDataType transformerDataType ->
  Double ->
  Tensor requiresGradient layout device dataType shape ->
  m output
mkTransformerAttentionMask transformerDataType attentionMaskBias paddingMask = do
  pmLayout <- sLayout paddingMask
  pmDevice <- sDevice paddingMask
  pmShape <- sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let emptyMask = sZeros SWithoutGradient pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
  pure $ maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

type MkTransformerDecoderAttentionMaskC m transformerDataType layout device shape seqDim output =
  ( MonadFail m,
    SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          'WithoutGradient
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
  forall m transformerDataType requiresGradient layout device dataType shape seqDim output.
  MkTransformerDecoderAttentionMaskC m transformerDataType layout device shape seqDim output =>
  SDataType transformerDataType ->
  Double ->
  Tensor requiresGradient layout device dataType shape ->
  m output
mkTransformerDecoderAttentionMask transformerDataType attentionMaskBias paddingMask = do
  pmLayout <- sLayout paddingMask
  pmDevice <- sDevice paddingMask
  pmShape <- sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let causalMask =
        unsqueeze @('SelectDim ('ByIndex 0))
          . bool
          . triu 1
          $ sOnes SWithoutGradient pmLayout pmDevice transformerDataType (SShape $ pmSeqDim :|: pmSeqDim :|: SNil)
      emptyMask = sZeros SWithoutGradient pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: pmSeqDim :|: pmSeqDim :|: SNil)
      booleanMask = causalMask `logicalOr` unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
  pure $
    maskedFill
      booleanMask
      attentionMaskBias
      emptyMask

type MkTransformerCrossAttentionMaskC m transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output =
  ( MonadFail m,
    SGetLayout layout,
    SGetDevice device,
    SGetShape shape,
    seqDim ~ (shape ! 1),
    output
      ~ Tensor
          (Seq (requiresGradient <+> 'WithoutGradient) 'WithoutGradient)
          (layout <+> 'Layout 'Dense)
          device
          (Seq (dataType <+> 'DataType 'Bool) transformerDataType)
          ( BroadcastShapesF
              (UnsqueezeF ('SelectDim ('ByIndex 1)) shape)
              ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim])
          )
  )

mkTransformerCrossAttentionMask ::
  forall m transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output.
  MkTransformerCrossAttentionMaskC m transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output =>
  SDataType transformerDataType ->
  SDim seqDim' ->
  Double ->
  Tensor requiresGradient layout device dataType shape ->
  m output
mkTransformerCrossAttentionMask transformerDataType seqDim' attentionMaskBias paddingMask = do
  pmLayout <- sLayout paddingMask
  pmDevice <- sDevice paddingMask
  pmShape <- sShape paddingMask
  pmSeqDim <- sGetDim (SSelectDim $ SByIndex @1) pmShape
  let emptyMask = sZeros SWithoutGradient pmLayout pmDevice transformerDataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim' :|: pmSeqDim :|: SNil)
  pure $ maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

data ShiftRight fillValue where
  ShiftRight :: forall fillValue. fillValue -> ShiftRight fillValue

instance HasInitialize (ShiftRight fillValue) fillValue generator generator where
  initialize fillValue = (ShiftRight fillValue,)

instance
  ( input
      ~ Tensor
          inputRequiresGradient
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
    filler
      ~ Tensor
          'WithoutGradient
          inputLayout
          inputDevice
          inputDataType
          fillerShape,
    fillerShape ~ 'Shape '[inputBatchDim, 'Dim ('Name "*") ('Size 1)],
    KnownLayout inputLayout,
    KnownDevice inputDevice,
    KnownDataType inputDataType,
    KnownShape inputShape,
    Scalar fillValue,
    rightShiftedInput
      ~ Tensor
          (inputRequiresGradient <|> 'WithoutGradient)
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
    inputLayout <- sLayout input
    inputDevice <- sDevice input
    inputDataType <- sDataType input
    inputShape <- sShape input
    inputBatchDim <- sGetDim (SSelectDim $ SByIndex @0) inputShape
    let filler = sFull SWithoutGradient inputLayout inputDevice inputDataType (SShape $ inputBatchDim :|: SName @"*" :&: SSize @1 :|: SNil) fillValue
    pure (cat @('SelectDim ('ByIndex 1)) (filler :. input :. HNil), g)
