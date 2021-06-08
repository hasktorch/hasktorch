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

module Torch.GraduallyTyped.NN.Transformer.BART.Common where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.Reader (ReaderT (runReaderT))
import Data.Coerce (Coercible, coerce)
import Data.Kind (Type)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import GHC.TypeNats (type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), KnownDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), KnownDevice (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (KnownLayout, Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasLookupDecoderStack)
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..), SequenceToSequenceTransformerWithLMHead, lookupSequenceToSequenceTransformer, lookupSequenceToSequenceTransformerWithLMHead)
import Torch.GraduallyTyped.NN.Transformer.Stack (HasLookupStack)
import Torch.GraduallyTyped.NN.Transformer.Type (MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, ShiftRight, TensorDict, TransformerStyle (BART), mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask, tensorDictFromPretrained)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF, type (!))
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), KnownShape (..), Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Creation (arangeNaturals)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, device, shape)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | BART dType.
type BARTDType = 'Float

-- | BART data type.
type BARTDataType = 'DataType BARTDType

-- | BART dropout probability type.
type BARTDropoutP = Float

-- | BART dropout rate.
-- 'dropout_rate = 0.1'
bartDropoutP :: BARTDropoutP
bartDropoutP = 0.1

-- | BART positional encoding dimension.
type BARTPosEncDim = 'Dim ('Name "*") ('Size 1026)

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

-- | BART model.
newtype
  BARTModel
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  BARTModel ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    BARTModelSeqToSeqF BARTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    BARTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

-- | BART model with language modelling head.
newtype
  BARTModelWithLMHead
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  BARTModelWithLMHead ::
    forall numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    BARTModelSeqToSeqF BARTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim ->
    BARTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

type family
  BARTModelSeqToSeqF
    ( bartModel ::
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
  BARTModelSeqToSeqF BARTModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformer
      numLayers
      numLayers
      'BART
      device
      BARTDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      BARTPosEncDim
      vocabDim
      BARTDropoutP
  BARTModelSeqToSeqF BARTModelWithLMHead numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    SequenceToSequenceTransformerWithLMHead
      numLayers
      numLayers
      'BART
      device
      BARTDataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      BARTPosEncDim
      vocabDim
      BARTDropoutP

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'BART ('Device 'CPU) BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'BART ('Device 'CPU) BARTDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim BARTDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (BARTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (BARTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (BARTModel numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        BARTModel <$> lookupSequenceToSequenceTransformer bartDropoutP bartEps "model."

instance
  ( KnownDim headDim,
    KnownDim headEmbedDim,
    KnownDim embedDim,
    KnownDim ffnDim,
    KnownDim inputEmbedDim,
    KnownDim vocabDim,
    HasLookupStack numLayers (1 <=? numLayers) numLayers 'BART ('Device 'CPU) BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTDropoutP (ReaderT TensorDict IO),
    HasLookupDecoderStack numLayers (1 <=? numLayers) numLayers 'BART ('Device 'CPU) BARTDataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim BARTDropoutP (ReaderT TensorDict IO)
  ) =>
  HasInitialize (BARTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  where
  type
    InitializeF (BARTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim) =
      FilePath -> IO (BARTModelWithLMHead numLayers ('Device 'CPU) headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  initialize filePath =
    do
      tensorDict <- tensorDictFromPretrained filePath
      flip runReaderT tensorDict $
        BARTModelWithLMHead <$> lookupSequenceToSequenceTransformerWithLMHead bartDropoutP bartEps ""

mkBARTInput ::
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
mkBARTInput = mkTransformerInput @batchDim @seqDim @m bartPadTokenId

mkBARTPaddingMask ::
  Tensor requiresGradient layout device dataType shape ->
  Tensor
    'WithoutGradient
    (layout <+> 'Layout 'Dense)
    (device <+> 'Device 'CPU)
    (Seq (dataType <+> 'DataType 'Int64) ('DataType 'Bool))
    (BroadcastShapesF shape ('Shape '[ 'Dim ('Name "*") ('Size 1)]))
mkBARTPaddingMask = mkTransformerPaddingMask bartPadTokenId

data BARTInput input decoderInput where
  BARTInput ::
    forall input decoderInput.
    { bartInput :: input,
      bartDecoderInput :: decoderInput
    } ->
    BARTInput input decoderInput

deriving instance
  ( Show input,
    Show decoderInput
  ) =>
  Show (BARTInput input decoderInput)

data BARTOutput decoderOutput encoderOutput inputPaddingMask where
  BARTOutput ::
    forall decoderOutput encoderOutput inputPaddingMask.
    { bartDecoderOutput :: decoderOutput,
      bartEncoderOutput :: encoderOutput,
      bartInputPaddingMask :: inputPaddingMask
    } ->
    BARTOutput decoderOutput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderOutput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (BARTOutput decoderOutput encoderOutput inputPaddingMask)

data BARTGenerationInput decoderInput encoderOutput inputPaddingMask where
  BARTGenerationInput ::
    forall decoderInput encoderOutput inputPaddingMask.
    { bartGenerationDecoderInput :: decoderInput,
      bartGenerationEncoderOutput :: encoderOutput,
      bartGenerationInputPaddingMask :: inputPaddingMask
    } ->
    BARTGenerationInput decoderInput encoderOutput inputPaddingMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show inputPaddingMask
  ) =>
  Show (BARTGenerationInput decoderInput encoderOutput inputPaddingMask)

-- | 'HasForward' instance for BART models.
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
    KnownDevice inputDevice,
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
    pos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          inputDevice
          ('DataType 'Int64)
          ('Shape '[inputSeqDim]),
    WithDimC inputSeqDim pos,
    WithDeviceC inputDevice (WithDimF inputSeqDim pos),
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
    rightShiftedDecoderInputPaddingMaskSeqDim ~ (rightShiftedDecoderInputPaddingMaskShape ! 1),
    decoderPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          rightShiftedDecoderInputDevice
          ('DataType 'Int64)
          ('Shape '[rightShiftedDecoderInputSeqDim]),
    WithDimC rightShiftedDecoderInputSeqDim decoderPos,
    WithDeviceC rightShiftedDecoderInputDevice (WithDimF rightShiftedDecoderInputSeqDim decoderPos),
    MkTransformerAttentionMaskC BARTDType BARTDataType inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC BARTDType BARTDataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC BARTDType BARTDataType rightShiftedDecoderInputPaddingMaskRequiresGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (BARTInput input decoderInput)
    generator
    (BARTOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward bartModel BARTInput {..} =
    let inputPaddingMask = mkBARTPaddingMask bartInput
        attentionMask =
          mkTransformerAttentionMask
            @BARTDType
            @BARTDataType
            @inputPaddingMaskRequiresGradient
            @inputPaddingMaskLayout
            @inputPaddingMaskDevice
            @inputPaddingMaskDataType
            @inputPaddingMaskShape
            bartAttentionMaskBias
            inputPaddingMask
        inputDevice = device bartInput
        [_, inputSeqDim] = shape bartInput
        pos =
          flip addScalar (2 :: Int) $
            withoutDim @inputSeqDim @pos
              ( withoutDevice @inputDevice
                  ( arangeNaturals
                      @'WithoutGradient
                      @('Layout 'Dense)
                      @inputDevice
                      @('DataType 'Int64)
                      @inputSeqDim
                  )
                  inputDevice
              )
              inputSeqDim
     in runIxState $
          ireturn bartDecoderInput
            >>>= IxState . forward (initialize @(ShiftRight Int) bartEOSTokenId)
            >>>= ( \rightShiftedDecoderInput ->
                     let rightShiftedDecoderInputDevice = device rightShiftedDecoderInput
                         [_, rightShiftedDecoderInputSeqDim] = shape rightShiftedDecoderInput
                         decoderPos =
                           flip addScalar (2 :: Int) $
                             withoutDim @rightShiftedDecoderInputSeqDim @decoderPos
                               ( withoutDevice @rightShiftedDecoderInputDevice
                                   ( arangeNaturals
                                       @'WithoutGradient
                                       @('Layout 'Dense)
                                       @rightShiftedDecoderInputDevice
                                       @('DataType 'Int64)
                                       @rightShiftedDecoderInputSeqDim
                                   )
                                   rightShiftedDecoderInputDevice
                               )
                               rightShiftedDecoderInputSeqDim
                         crossAttentionMask =
                           withoutDim @rightShiftedDecoderInputSeqDim @(inputPaddingMask -> crossAttentionMask)
                             ( mkTransformerCrossAttentionMask
                                 @BARTDType
                                 @BARTDataType
                                 @rightShiftedDecoderInputSeqDim
                                 @inputPaddingMaskRequiresGradient
                                 @inputPaddingMaskLayout
                                 @inputPaddingMaskDevice
                                 @inputPaddingMaskDataType
                                 @inputPaddingMaskShape
                                 bartAttentionMaskBias
                             )
                             rightShiftedDecoderInputSeqDim
                             inputPaddingMask
                      in ireturn (mkBARTPaddingMask bartDecoderInput)
                           >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          mkTransformerDecoderAttentionMask
                                            @BARTDType
                                            @BARTDataType
                                            @rightShiftedDecoderInputPaddingMaskRequiresGradient
                                            @rightShiftedDecoderInputPaddingMaskLayout
                                            @rightShiftedDecoderInputPaddingMaskDevice
                                            @rightShiftedDecoderInputPaddingMaskDataType
                                            @rightShiftedDecoderInputPaddingMaskShape
                                            bartAttentionMaskBias
                                            rightShiftedDecoderInputPaddingMask
                                     in ireturn (SequenceToSequenceTransformerInput bartInput rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
                                )
                           >>>= IxState . forward (coerce bartModel :: BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                           >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ BARTOutput decoderOutput encoderOutput inputPaddingMask
                                )
                 )

-- | 'HasForward' instance for BART models.
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
    KnownDevice rightShiftedDecoderInputDevice,
    rightShiftedDecoderInputSeqDim ~ (rightShiftedDecoderInputShape ! 1),
    KnownDim rightShiftedDecoderInputSeqDim,
    KnownShape rightShiftedDecoderInputShape,
    decoderPos
      ~ Tensor
          'WithoutGradient
          ('Layout 'Dense)
          rightShiftedDecoderInputDevice
          ('DataType 'Int64)
          ('Shape '[rightShiftedDecoderInputSeqDim]),
    WithDimC rightShiftedDecoderInputSeqDim decoderPos,
    WithDeviceC rightShiftedDecoderInputDevice (WithDimF rightShiftedDecoderInputSeqDim decoderPos),
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
    MkTransformerCrossAttentionMaskC BARTDType BARTDataType rightShiftedDecoderInputSeqDim inputPaddingMaskRequiresGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC BARTDType BARTDataType rightShiftedDecoderInputPaddingMaskRequiresGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput,
    Coercible
      (BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
      (bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
  ) =>
  HasForward
    (bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (BARTGenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (BARTOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward bartModel BARTGenerationInput {..} =
    runIxState $
      ireturn bartGenerationDecoderInput
        >>>= IxState . forward (initialize @(ShiftRight Int) bartEOSTokenId)
        >>>= ( \rightShiftedDecoderInput ->
                 let rightShiftedDecoderInputDevice = device rightShiftedDecoderInput
                     [_, rightShiftedDecoderInputSeqDim] = shape rightShiftedDecoderInput
                     decoderPos =
                       flip addScalar (2 :: Int) $
                         withoutDim @rightShiftedDecoderInputSeqDim @decoderPos
                           ( withoutDevice @rightShiftedDecoderInputDevice
                               ( arangeNaturals
                                   @'WithoutGradient
                                   @('Layout 'Dense)
                                   @rightShiftedDecoderInputDevice
                                   @('DataType 'Int64)
                                   @rightShiftedDecoderInputSeqDim
                               )
                               rightShiftedDecoderInputDevice
                           )
                           rightShiftedDecoderInputSeqDim
                     crossAttentionMask =
                       withoutDim @rightShiftedDecoderInputSeqDim @(inputPaddingMask -> crossAttentionMask)
                         ( mkTransformerCrossAttentionMask
                             @BARTDType
                             @BARTDataType
                             @rightShiftedDecoderInputSeqDim
                             @inputPaddingMaskRequiresGradient
                             @inputPaddingMaskLayout
                             @inputPaddingMaskDevice
                             @inputPaddingMaskDataType
                             @inputPaddingMaskShape
                             bartAttentionMaskBias
                         )
                         rightShiftedDecoderInputSeqDim
                         bartGenerationInputPaddingMask
                  in ireturn (mkBARTPaddingMask bartGenerationDecoderInput)
                       >>>= IxState . forward (initialize @(ShiftRight Int) 0)
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      mkTransformerDecoderAttentionMask
                                        @BARTDType
                                        @BARTDataType
                                        @rightShiftedDecoderInputPaddingMaskRequiresGradient
                                        @rightShiftedDecoderInputPaddingMaskLayout
                                        @rightShiftedDecoderInputPaddingMaskDevice
                                        @rightShiftedDecoderInputPaddingMaskDataType
                                        @rightShiftedDecoderInputPaddingMaskShape
                                        bartAttentionMaskBias
                                        rightShiftedDecoderInputPaddingMask
                                 in ireturn (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput bartGenerationEncoderOutput decoderPos decoderAttentionMask crossAttentionMask)
                            )
                       >>>= IxState . forward (coerce bartModel :: BARTModelSeqToSeqF bartModel numLayers device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ BARTOutput decoderOutput encoderOutput bartGenerationInputPaddingMask
                            )
             )
