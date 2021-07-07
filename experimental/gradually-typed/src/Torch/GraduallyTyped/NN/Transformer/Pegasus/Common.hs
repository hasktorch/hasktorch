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

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import System.IO.Unsafe (unsafePerformIO)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkPosC, MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, MkTransformerPaddingMaskC, ShiftRight, TransformerHead (..), TransformerStyle (Pegasus), mkPos, mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), SDim, Shape (..), Size (..))
import Torch.GraduallyTyped.Tensor.Type (SGetDim, Tensor, sShape)
import Torch.GraduallyTyped.Unify (type (<+>))

-- | Pegasus dType.
type PegasusDType = 'Float

-- | Pegasus dType singleton.
pegasusDType :: SDType PegasusDType
pegasusDType = sing @PegasusDType

-- | Pegasus data type.
type PegasusDataType = 'DataType PegasusDType

-- | Pegasus data type singleton.
pegasusDataType :: SDataType PegasusDataType
pegasusDataType = sing @PegasusDataType

-- | Pegasus dropout probability type.
type PegasusDropoutP = Float

-- | Pegasus dropout rate.
-- 'dropout_rate = 0.1'
pegasusDropoutP :: PegasusDropoutP
pegasusDropoutP = 0.1

-- | Pegasus positional encoding dimension.
type PegasusPosEncDim = 'Dim ('Name "*") ('Size 512)

-- | Pegasus positional encoding dimension singleton.
pegasusPosEncDim :: SDim PegasusPosEncDim
pegasusPosEncDim = sing @PegasusPosEncDim

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

data
  GPegasusModel
    (pegasusModel :: Type)
  where
  GPegasusModel ::
    forall pegasusModel.
    { pegasusModel :: pegasusModel,
      pegasusShiftRightDecoderInput :: ShiftRight Int,
      pegasusShiftRightPaddingMask :: ShiftRight Int
    } ->
    GPegasusModel pegasusModel

-- | Pegasus model.
data
  PegasusModel
    (transformerHead :: TransformerHead)
    (numLayers :: Nat)
    (gradient :: Gradient RequiresGradient)
    (device :: Device (DeviceType Nat))
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
  where
  PegasusModel ::
    forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    GPegasusModel
      (SequenceToSequenceTransformer 'Pegasus transformerHead numLayers numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim vocabDim PegasusDropoutP) ->
    PegasusModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

instance
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim,
    HasStateDict
      (SequenceToSequenceTransformer 'Pegasus transformerHead numLayers numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim vocabDim PegasusDropoutP)
      (SGradient gradient, SDevice device, SDataType PegasusDataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim PegasusPosEncDim, SDim vocabDim, PegasusDropoutP, Double)
  ) =>
  HasStateDict
    (PegasusModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (SGradient gradient, SDevice device)
  where
  fromStateDict (gradient, device) k =
    let headDim = sing @headDim
        headEmbedDim = sing @headEmbedDim
        embedDim = sing @embedDim
        inputEmbedDim = sing @inputEmbedDim
        ffnDim = sing @ffnDim
        vocabDim = sing @vocabDim
     in PegasusModel
          <$> ( GPegasusModel
                  <$> fromStateDict (gradient, device, pegasusDataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, pegasusPosEncDim, vocabDim, pegasusDropoutP, pegasusEps) (k <> "model.")
                  <*> fromStateDict pegasusBOSTokenId k
                  <*> fromStateDict 0 k
              )
  toStateDict k (PegasusModel GPegasusModel {..}) = do
    toStateDict (k <> "model.") pegasusModel
    toStateDict k pegasusShiftRightDecoderInput
    toStateDict k pegasusShiftRightPaddingMask

mkPegasusInput ::
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
mkPegasusInput = mkTransformerInput pegasusPadTokenId

mkPegasusPaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  Tensor gradient layout device dataType shape ->
  output
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

-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
instance
  ( input ~ Tensor inputGradient inputLayout inputDevice inputDataType inputShape,
    MkPosC inputDevice inputShape inputSeqDim inputSeqName inputSeqSize pos,
    MkTransformerPaddingMaskC inputLayout inputDevice inputDataType inputShape inputPaddingMask,
    inputPaddingMask ~ Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape,
    decoderInput ~ Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
    rightShiftedDecoderInput ~ Tensor rightShiftedDecoderInputGradient rightShiftedDecoderInputLayout rightShiftedDecoderInputDevice rightShiftedDecoderInputDataType rightShiftedDecoderInputShape,
    MkPosC rightShiftedDecoderInputDevice rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize decoderPos,
    MkTransformerPaddingMaskC decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape decoderInputPaddingMask,
    rightShiftedDecoderInputPaddingMask ~ Tensor rightShiftedDecoderInputPaddingMaskGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape,
    MkTransformerAttentionMaskC PegasusDataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC PegasusDataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC PegasusDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (SequenceToSequenceTransformer 'Pegasus transformerHead numLayers numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim vocabDim PegasusDropoutP)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (PegasusModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (PegasusInput input decoderInput)
    generator
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (PegasusModel GPegasusModel {..}) PegasusInput {..} =
    let inputPaddingMask = mkPegasusPaddingMask pegasusInput
        attentionMask = unsafePerformIO $ mkTransformerAttentionMask pegasusDataType pegasusAttentionMaskBias inputPaddingMask
        pos = unsafePerformIO $ mkPos pegasusInput
     in runIxState $
          ireturn pegasusDecoderInput
            >>>= IxState . forward pegasusShiftRightDecoderInput
            >>>= ( \rightShiftedDecoderInput ->
                     let decoderPos = unsafePerformIO $ mkPos rightShiftedDecoderInput
                         crossAttentionMask =
                           unsafePerformIO $
                             mkTransformerCrossAttentionMask
                               pegasusDataType
                               (sShape rightShiftedDecoderInput)
                               pegasusAttentionMaskBias
                               inputPaddingMask
                      in ireturn (mkPegasusPaddingMask pegasusDecoderInput)
                           >>>= IxState . forward pegasusShiftRightPaddingMask
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          unsafePerformIO $
                                            mkTransformerDecoderAttentionMask
                                              pegasusDataType
                                              pegasusAttentionMaskBias
                                              rightShiftedDecoderInputPaddingMask
                                     in ireturn (SequenceToSequenceTransformerInput pegasusInput rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
                                )
                           >>>= IxState . forward pegasusModel
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
  ( inputPaddingMask ~ Tensor inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape,
    decoderInput ~ Tensor decoderInputGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
    rightShiftedDecoderInput ~ Tensor rightShiftedDecoderInputGradient rightShiftedDecoderInputLayout rightShiftedDecoderInputDevice rightShiftedDecoderInputDataType rightShiftedDecoderInputShape,
    MkPosC rightShiftedDecoderInputDevice rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim rightShiftedDecoderInputSeqName rightShiftedDecoderInputSeqSize decoderPos,
    MkTransformerPaddingMaskC decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape decoderInputPaddingMask,
    rightShiftedDecoderInputPaddingMask ~ Tensor rightShiftedDecoderInputPaddingMaskGradient rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskDataType rightShiftedDecoderInputPaddingMaskShape,
    MkTransformerAttentionMaskC PegasusDataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC PegasusDataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC PegasusDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (SequenceToSequenceTransformer 'Pegasus transformerHead numLayers numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim vocabDim PegasusDropoutP)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (PegasusModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (PegasusGenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (PegasusModel GPegasusModel {..}) PegasusGenerationInput {..} =
    runIxState $
      ireturn pegasusGenerationDecoderInput
        >>>= IxState . forward pegasusShiftRightDecoderInput
        >>>= ( \rightShiftedDecoderInput ->
                 let decoderPos = unsafePerformIO $ mkPos rightShiftedDecoderInput
                     crossAttentionMask =
                       unsafePerformIO $
                         mkTransformerCrossAttentionMask
                           pegasusDataType
                           (sShape rightShiftedDecoderInput)
                           pegasusAttentionMaskBias
                           pegasusGenerationInputPaddingMask
                  in ireturn (mkPegasusPaddingMask pegasusGenerationDecoderInput)
                       >>>= IxState . forward pegasusShiftRightPaddingMask
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      unsafePerformIO $
                                        mkTransformerDecoderAttentionMask
                                          pegasusDataType
                                          pegasusAttentionMaskBias
                                          rightShiftedDecoderInputPaddingMask
                                 in ireturn (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput pegasusGenerationEncoderOutput decoderPos decoderAttentionMask crossAttentionMask)
                            )
                       >>>= IxState . forward pegasusModel
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ PegasusOutput decoderOutput encoderOutput pegasusGenerationInputPaddingMask
                            )
             )
