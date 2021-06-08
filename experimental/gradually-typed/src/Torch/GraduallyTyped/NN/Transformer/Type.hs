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
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TemplateHaskell #-}
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
import Data.Singletons.TH (genSingletons)
import Foreign.ForeignPtr (ForeignPtr)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..))
import Torch.GraduallyTyped.Layout (KnownLayout (..), Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape.Class (AddDimF, BroadcastShapesF, ReplaceDimF, type (!))
import Torch.GraduallyTyped.Shape.Type (By (..), Dim (..), KnownDim (..), KnownShape (..), Name (..), SelectDim (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor (cat)
import Torch.GraduallyTyped.Tensor.Creation (WithCreateC (..), full, ones, zeros)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (UnsqueezeF, unsqueeze)
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

data TransformerStyle = T5 | BART | MBART | BERT | RoBERTa | Pegasus | GPT2
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
  Int ->
  WithDimF batchDim (WithDimF seqDim ([[Int]] -> m output))
mkTransformerInput padTokenId =
  withDim @batchDim $
    \(Dim batchName batchSize) ->
      withDim @seqDim @([[Int]] -> m output) $
        \(Dim seqName seqSize) xs -> do
          let emptySeq = replicate (fromIntegral seqSize) padTokenId
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
        full
          @'WithoutGradient
          @('Layout 'Dense)
          @('Device 'CPU)
          @('DataType 'Int64)
          @('Shape '[ 'Dim ('Name "*") ('Size 1)])
          padTokenId
   in input ==. padToken

type MkTransformerAttentionMaskC transformerDType transformerDataType requiresGradient layout device dataType shape seqDim output =
  ( KnownDType transformerDType,
    KnownLayout layout,
    KnownDevice device,
    KnownShape shape,
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
          ),
    WithCreateC (Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])) 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
  )

mkTransformerAttentionMask ::
  forall transformerDType transformerDataType requiresGradient layout device dataType shape seqDim output.
  MkTransformerAttentionMaskC transformerDType transformerDataType requiresGradient layout device dataType shape seqDim output =>
  Double ->
  Tensor requiresGradient layout device dataType shape ->
  output
mkTransformerAttentionMask attentionMaskBias paddingMask =
  let layoutType = layout paddingMask
      deviceType = device paddingMask
      dType = dTypeVal @transformerDType
      [_batchDim, seqDim] = shape paddingMask
      emptyMask =
        withoutCreate @(Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])) @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          (zeros @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim]))
          WithoutGradient
          layoutType
          deviceType
          dType
          [Dim "*" 1, seqDim, seqDim]
   in maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

type MkTransformerDecoderAttentionMaskC transformerDType transformerDataType (requiresGradient :: RequiresGradient) layout device dataType shape seqDim output =
  ( KnownDType transformerDType,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape,
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
          ),
    WithCreateC (Tensor 'WithoutGradient layout device transformerDataType ('Shape '[seqDim, seqDim])) 'WithoutGradient layout device transformerDataType ('Shape '[seqDim, seqDim]),
    WithCreateC (Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])) 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
  )

mkTransformerDecoderAttentionMask ::
  forall transformerDType transformerDataType requiresGradient layout device dataType shape seqDim output.
  MkTransformerDecoderAttentionMaskC transformerDType transformerDataType requiresGradient layout device dataType shape seqDim output =>
  Double ->
  Tensor requiresGradient layout device dataType shape ->
  output
mkTransformerDecoderAttentionMask attentionMaskBias paddingMask =
  let layoutType = layout paddingMask
      deviceType = device paddingMask
      dType' = dTypeVal @transformerDType
      [_batchDim, seqDim] = shape paddingMask
      causalMask =
        unsqueeze @('SelectDim ('ByIndex 0))
          . bool
          . triu 1
          $ withoutCreate @(Tensor 'WithoutGradient layout device transformerDataType ('Shape '[seqDim, seqDim])) @'WithoutGradient @layout @device @transformerDataType @('Shape '[seqDim, seqDim])
            (ones @'WithoutGradient @layout @device @transformerDataType @('Shape '[seqDim, seqDim]))
            WithoutGradient
            layoutType
            deviceType
            dType'
            [seqDim, seqDim]
      emptyMask =
        withoutCreate @(Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])) @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim])
          (zeros @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim, seqDim]))
          WithoutGradient
          layoutType
          deviceType
          dType'
          [Dim "*" 1, seqDim, seqDim]
      booleanMask = causalMask `logicalOr` unsqueeze @('SelectDim ('ByIndex 1)) paddingMask
   in maskedFill
        booleanMask
        attentionMaskBias
        emptyMask

type MkTransformerCrossAttentionMaskC transformerDType transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output =
  ( KnownDType transformerDType,
    KnownLayout layout,
    KnownDevice device,
    KnownDataType dataType,
    KnownShape shape,
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
          ),
    WithCreateC (Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim])) 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim]),
    WithDimC seqDim' (Tensor requiresGradient layout device dataType shape -> output)
  )

mkTransformerCrossAttentionMask ::
  forall transformerDType transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output.
  MkTransformerCrossAttentionMaskC transformerDType transformerDataType seqDim' requiresGradient layout device dataType shape seqDim output =>
  Double ->
  WithDimF seqDim' (Tensor requiresGradient layout device dataType shape -> output)
mkTransformerCrossAttentionMask attentionMaskBias =
  withDim @seqDim' @(Tensor requiresGradient layout device dataType shape -> output) $
    \seqDim' paddingMask ->
      let layoutType = layout paddingMask
          deviceType = device paddingMask
          dType = dTypeVal @transformerDType
          [_batchDim, seqDim] = shape paddingMask
          emptyMask =
            withoutCreate @(Tensor 'WithoutGradient layout device transformerDataType ('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim])) @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim])
              (zeros @'WithoutGradient @layout @device @transformerDataType @('Shape '[ 'Dim ('Name "*") ('Size 1), seqDim', seqDim]))
              WithoutGradient
              layoutType
              deviceType
              dType
              [Dim "*" 1, seqDim', seqDim]
       in maskedFill (unsqueeze @('SelectDim ('ByIndex 1)) paddingMask) attentionMaskBias emptyMask

data ShiftRight fillValue where
  ShiftRight :: forall fillValue. fillValue -> ShiftRight fillValue

instance HasInitialize (ShiftRight fillValue) where
  type InitializeF (ShiftRight fillValue) = fillValue -> ShiftRight fillValue
  initialize fillValue = ShiftRight fillValue

instance
  ( input
      ~ Tensor
          inputRequiresGradient
          inputLayout
          inputDevice
          inputDataType
          inputShape,
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
    WithCreateC (fillValue -> filler) 'WithoutGradient inputLayout inputDevice inputDataType fillerShape,
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
  forward (ShiftRight fillValue) input g =
    let inputLayoutType = layout input
        inputDeviceType = device input
        inputDType = dataType input
        inputBatchDim : _ = shape input
        fillerDims = [inputBatchDim, Dim "*" 1]
        filler =
          withoutCreate @(fillValue -> filler) @'WithoutGradient @inputLayout @inputDevice @inputDataType @fillerShape
            (full @'WithoutGradient @inputLayout @inputDevice @inputDataType @fillerShape @fillValue)
            WithoutGradient
            inputLayoutType
            inputDeviceType
            inputDType
            fillerDims
            fillValue
     in (cat @('SelectDim ('ByIndex 1)) (filler :. input :. HNil), g)
