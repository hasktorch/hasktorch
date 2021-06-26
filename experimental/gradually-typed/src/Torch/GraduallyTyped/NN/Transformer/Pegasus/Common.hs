{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Coerce (Coercible, coerce)
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeNats (type (<=?))
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..))
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..), SequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformer, lookupSequenceToSequenceTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, ShiftRight, TensorDict, TransformerStyle (Pegasus), mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, sGetDim, type (!))
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), KnownShape (..), Name (..), SBy (..), SDim, SSelectDim (..), Shape (..), Size (..), sDimSize)
import Torch.GraduallyTyped.Tensor.Creation (sArangeNaturals)
import Torch.GraduallyTyped.Tensor.Type (SGetDevice, SGetLayout, SGetShape, Tensor, sDevice, sShape)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Pegasus dType.
type PegasusDType = 'Float

-- | Pegasus dType.
pegasusDType :: SDType PegasusDType
pegasusDType = SFloat

-- | Pegasus data type.
type PegasusDataType = 'DataType PegasusDType

-- | Pegasus data type.
pegasusDataType :: SDataType PegasusDataType
pegasusDataType = SDataType pegasusDType

-- | Pegasus dropout probability type.
type PegasusDropoutP = Float

-- | Pegasus dropout rate.
-- 'dropout_rate = 0.1'
pegasusDropoutP :: PegasusDropoutP
pegasusDropoutP = 0.1

-- | Pegasus positional encoding dimension.
type PegasusPosEncDim = 'Dim ('Name "*") ('Size 512)

-- | Pegasus layer-norm epsilon.
pegasusEps :: Double
pegasusEps = 1e-5

-- | Pegasus maximum number of position embeddings.
-- 'max_position_embeddings = 512'
pegasusMaxPositionEmbeddings :: Int
pegasusMaxPositionEmbeddings = 512

-- | Pegasus padding token id.
-- 'pad_token_id = 0'
pegasusPadTokenId :: Int
pegasusPadTokenId = 0

-- | Pegasus begin-of-sentence token id.
-- 'bos_token_id = 0'
pegasusBOSTokenId :: Int
pegasusBOSTokenId = pegasusPadTokenId

-- | Pegasus end-of-sentence token id.
-- 'eos_token_id = 0'
pegasusEOSTokenId :: Int
pegasusEOSTokenId = 1

-- | Pegasus attention mask bias
pegasusAttentionMaskBias :: Double
pegasusAttentionMaskBias = -10000

-- | Pegasus model.
newtype
  PegasusModel
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  PegasusModel ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    PegasusModelSeqToSeqF PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

-- | Pegasus model with language modelling head.
newtype
  PegasusModelWithLMHead
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  PegasusModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    PegasusModelSeqToSeqF PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

type family
  PegasusModelSeqToSeqF
    ( pegasusModel ::
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
  PegasusModelSeqToSeqF PegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer
      numLayers
      numLayers
      'Pegasus
      device
      PegasusDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      PegasusPosEncDim
      vocabDim
      PegasusDropoutP
  PegasusModelSeqToSeqF PegasusModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformerWithLMHead
      numLayers
      numLayers
      'Pegasus
      device
      PegasusDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      PegasusPosEncDim
      vocabDim
      PegasusDropoutP

instance
  ( KnownDim headDim,
    SingI headDim,
    SingI headEmbedDim,
    KnownDim embedDim,
    SingI embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    SingI inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (PegasusModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        PegasusModel <$> lookupSequenceToSequenceTransformer headDim headEmbedDim embedDim inputEmbedDim pegasusDropoutP pegasusEps "model."

instance
  ( KnownDim headDim,
    SingI headDim,
    SingI headEmbedDim,
    KnownDim embedDim,
    SingI embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    SingI inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'Pegasus ('Device 'CPU) PegasusDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim PegasusDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (PegasusModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      let headDim = sing @headDim
          headEmbedDim = sing @headEmbedDim
          embedDim = sing @embedDim
          inputEmbedDim = sing @inputEmbedDim
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        PegasusModelWithLMHead <$> lookupSequenceToSequenceTransformerWithLMHead headDim headEmbedDim embedDim inputEmbedDim pegasusDropoutP pegasusEps ""

mkPegasusInput ::
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
  SDim batchDim ->
  SDim seqDim ->
  [[Int]] ->
  m output
mkPegasusInput = mkTransformerInput pegasusPadTokenId

mkPegasusPaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkPegasusPaddingMask = mkTransformerPaddingMask pegasusPadTokenId

data PegasusInput input decoderInput where
  PegasusInput ::
    forall input decoderInput.
    { pegasusInput :: input,
      pegasusDecoderInput :: decoderInput
    } ->
    PegasusInput input decoderInput

deriving instance
  ( Show input,
    Show decoderInput
  ) =>
  Show (PegasusInput input decoderInput)

data PegasusOutput decoderOutput encoderOutput inputPaddingMask where
  PegasusOutput ::
    forall decoderOutput encoderOutput inputPaddingMask.
    { pegasusDecoderOutput :: decoderOutput,
      pegasusEncoderOutput :: encoderOutput,
      pegasusInputPaddingMask :: inputPaddingMask
    } ->
    PegasusOutput decoderOutput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderOutput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (PegasusOutput decoderOutput encoderOutput inputPaddingMask)

data PegasusGenerationInput decoderInput encoderOutput inputPaddingMask where
  PegasusGenerationInput ::
    forall decoderInput encoderOutput inputPaddingMask.
    { pegasusGenerationDecoderInput :: decoderInput,
      pegasusGenerationEncoderOutput :: encoderOutput,
      pegasusGenerationInputPaddingMask :: inputPaddingMask
    } ->
    PegasusGenerationInput decoderInput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (PegasusGenerationInput decoderInput encoderOutput inputPaddingMask)

-- | 'HasForward' instance for Pegasus models.
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
    SGetLayout inputLayout,
    SGetDevice inputDevice,
    SGetShape inputShape,
    inputSeqDim ~ (inputShape ! 1),
    inputSeqDim ~ 'Dim inputSeqName inputSeqSize,
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
    pos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          inputDevice
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") inputSeqSize]),
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
    KnownDevice rightShiftedDecoderInputDevice,
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
    SGetDevice rightShiftedDecoderInputDevice,
    SGetShape rightShiftedDecoderInputShape,
    rightShiftedDecoderInputPaddingMaskSeqDim ~ (rightShiftedDecoderInputPaddingMaskShape ! 1),
    rightShiftedDecoderInputSeqDim ~ 'Dim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize,
    decoderPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          rightShiftedDecoderInputDevice
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") rightShiftedDecoderInputSeqSize]),
    MkTransformerAttentionMaskC IO PegasusDataType inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC IO PegasusDataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC IO PegasusDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (PegasusInput input decoderInput)
    generator
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward pegasusModel PegasusInput {..} =
    let inputPaddingMask = mkPegasusPaddingMask pegasusInput
        attentionMask = unsafePerformIO $ mkTransformerAttentionMask pegasusDataType pegasusAttentionMaskBias inputPaddingMask
        inputDevice = unsafePerformIO $ sDevice pegasusInput
        inputShape = unsafePerformIO $ sShape pegasusInput
        inputSeqDim = unsafePerformIO $ sGetDim (SSelectDim $ SByIndex @1) inputShape
        inputSeqSize = sDimSize inputSeqDim
        pos =
          sArangeNaturals
            SWithoutGradient
            (SLayout SDense)
            inputDevice
            (SDataType SInt64)
            inputSeqSize
     in runIxState $
          ireturn pegasusDecoderInput
            >>>= IxState . forward (initialize @(ShiftRight Int) pegasusBOSTokenId)
            >>>= ( \rightShiftedDecoderInput ->
                     let rightShiftedDecoderInputDevice = unsafePerformIO $ sDevice rightShiftedDecoderInput
                         rightShiftedDecoderInputShape = unsafePerformIO $ sShape rightShiftedDecoderInput
                         rightShiftedDecoderInputSeqDim = unsafePerformIO $ sGetDim (SSelectDim $ SByIndex @1) rightShiftedDecoderInputShape
                         rightShiftedDecoderInputSeqSize = sDimSize rightShiftedDecoderInputSeqDim
                         decoderPos =
                           sArangeNaturals
                             SWithoutGradient
                             (SLayout SDense)
                             rightShiftedDecoderInputDevice
                             (SDataType SInt64)
                             rightShiftedDecoderInputSeqSize
                         crossAttentionMask =
                           unsafePerformIO $
                             mkTransformerCrossAttentionMask
                               pegasusDataType
                               rightShiftedDecoderInputSeqDim
                               pegasusAttentionMaskBias
                               inputPaddingMask
                      in ireturn (mkPegasusPaddingMask pegasusDecoderInput)
                           >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          unsafePerformIO $
                                            mkTransformerDecoderAttentionMask
                                              pegasusDataType
                                              pegasusAttentionMaskBias
                                              rightShiftedDecoderInputPaddingMask
                                     in ireturn (SequenceToSequenceTransformerInput pegasusInput rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
                                )
                           >>>= IxState . forward (coerce pegasusModel :: PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                           >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ PegasusOutput decoderOutput encoderOutput inputPaddingMask
                                )
                 )

-- | 'HasForward' instance for Pegasus models.
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
    SGetDevice rightShiftedDecoderInputDevice,
    SGetShape rightShiftedDecoderInputShape,
    rightShiftedDecoderInputSeqDim ~ (rightShiftedDecoderInputShape ! 1),
    rightShiftedDecoderInputSeqDim ~ 'Dim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize,
    decoderPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          rightShiftedDecoderInputDevice
          ('DataType 'Int64)
          ('Shape '[ 'Dim ('Name "*") rightShiftedDecoderInputSeqSize]),
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
    MkTransformerCrossAttentionMaskC IO PegasusDataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC IO PegasusDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (PegasusGenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward pegasusModel PegasusGenerationInput {..} =
    runIxState $
      ireturn pegasusGenerationDecoderInput
        >>>= IxState . forward (initialize @(ShiftRight Int) pegasusBOSTokenId)
        >>>= ( \rightShiftedDecoderInput ->
                 let rightShiftedDecoderInputDevice = unsafePerformIO $ sDevice rightShiftedDecoderInput
                     rightShiftedDecoderInputShape = unsafePerformIO $ sShape rightShiftedDecoderInput
                     rightShiftedDecoderInputSeqDim = unsafePerformIO $ sGetDim (SSelectDim $ SByIndex @1) rightShiftedDecoderInputShape
                     rightShiftedDecoderInputSeqSize = sDimSize rightShiftedDecoderInputSeqDim
                     decoderPos =
                       sArangeNaturals
                         SWithoutGradient
                         (SLayout SDense)
                         rightShiftedDecoderInputDevice
                         (SDataType SInt64)
                         rightShiftedDecoderInputSeqSize
                     crossAttentionMask =
                       unsafePerformIO $
                         mkTransformerCrossAttentionMask
                           pegasusDataType
                           rightShiftedDecoderInputSeqDim
                           pegasusAttentionMaskBias
                           pegasusGenerationInputPaddingMask
                  in ireturn (mkPegasusPaddingMask pegasusGenerationDecoderInput)
                       >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      unsafePerformIO $
                                        mkTransformerDecoderAttentionMask
                                          pegasusDataType
                                          pegasusAttentionMaskBias
                                          rightShiftedDecoderInputPaddingMask
                                 in ireturn (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput pegasusGenerationEncoderOutput decoderPos decoderAttentionMask crossAttentionMask)
                            )
                       >>>= IxState . forward (coerce pegasusModel :: PegasusModelSeqToSeqF pegasusModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ PegasusOutput decoderOutput encoderOutput pegasusGenerationInputPaddingMask
                            )
             )