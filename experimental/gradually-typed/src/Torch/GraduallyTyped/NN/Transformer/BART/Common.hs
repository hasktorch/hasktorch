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
{-# OPTIONS_GHC -v2 -Wall #-}

module Torch.GraduallyTyped.NN.Transformer.BART.Common where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.Prelude.List (SList (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice (..), SDeviceType (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..), SLayout (..), SLayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (initialize), HasStateDict (..))
import Torch.GraduallyTyped.NN.Transformer.SequenceToSequence (SequenceToSequenceTransformer, SequenceToSequenceTransformerGenerationInput (..), SequenceToSequenceTransformerInput (..), SequenceToSequenceTransformerOutput (..))
import Torch.GraduallyTyped.NN.Transformer.Type (MkPosC, MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, MkTransformerPaddingMaskC, ShiftRight (..), TransformerHead (..), TransformerStyle (BART), mkPos, mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask)
import Torch.GraduallyTyped.Random (sMkGenerator)
import Torch.GraduallyTyped.RequiresGradient (Gradient (..), RequiresGradient (..), SGradient (..), SRequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), KnownDim (..), Name (..), SDim (..), SName (..), SShape (..), SSize (..), Shape (..), Size (..), pattern (:&:), pattern (:|:))
import Torch.GraduallyTyped.Tensor.Creation (sOnes)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (addScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor, sShape)

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

-- | BART dropout probability type.
type BARTDropoutP = Float

-- | BART dropout rate.
-- 'dropout_rate = 0.1'
bartDropoutP :: BARTDropoutP
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

data
  GBARTModel
    (bartModel :: Type)
  where
  GBARTModel ::
    forall bartModel.
    { bartModel :: bartModel,
      bartShiftRightDecoderInput :: ShiftRight Int,
      bartShiftRightPaddingMask :: ShiftRight Int
    } ->
    GBARTModel bartModel

-- | BART model.
data
  BARTModel
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
  BARTModel ::
    forall transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim.
    GBARTModel
      (SequenceToSequenceTransformer 'BART transformerHead numLayers numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim vocabDim BARTDropoutP) ->
    BARTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim
  deriving stock (Generic)

instance
  ( SingI headDim,
    SingI headEmbedDim,
    SingI embedDim,
    SingI inputEmbedDim,
    SingI ffnDim,
    SingI vocabDim,
    HasStateDict
      (SequenceToSequenceTransformer 'BART transformerHead numLayers numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim vocabDim BARTDropoutP)
      (SGradient gradient, SDevice device, SDataType BARTDataType, SDim headDim, SDim headEmbedDim, SDim embedDim, SDim inputEmbedDim, SDim ffnDim, SDim BARTPosEncDim, SDim vocabDim, BARTDropoutP, Double)
  ) =>
  HasStateDict
    (BARTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (SGradient gradient, SDevice device)
  where
  fromStateDict (gradient, device) k =
    let headDim = sing @headDim
        headEmbedDim = sing @headEmbedDim
        embedDim = sing @embedDim
        inputEmbedDim = sing @inputEmbedDim
        ffnDim = sing @ffnDim
        vocabDim = sing @vocabDim
     in BARTModel
          <$> ( GBARTModel
                  <$> fromStateDict (gradient, device, bartDataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, bartPosEncDim, vocabDim, bartDropoutP, bartEps) k
                  <*> fromStateDict bartEOSTokenId k
                  <*> fromStateDict 0 k
              )
  toStateDict k (BARTModel GBARTModel {..}) = do
    toStateDict k bartModel
    toStateDict k bartShiftRightDecoderInput
    toStateDict k bartShiftRightPaddingMask

mkBARTInput ::
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
mkBARTInput = mkTransformerInput bartPadTokenId

mkBARTPaddingMask ::
  forall gradient layout device dataType shape output.
  MkTransformerPaddingMaskC layout device dataType shape output =>
  Tensor gradient layout device dataType shape ->
  output
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
    MkTransformerAttentionMaskC BARTDataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC BARTDataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC BARTDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (SequenceToSequenceTransformer 'BART transformerHead numLayers numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim vocabDim BARTDropoutP)
      (SequenceToSequenceTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (BARTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (BARTInput input decoderInput)
    generator
    (BARTOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (BARTModel GBARTModel {..}) BARTInput {..} =
    let inputPaddingMask = mkBARTPaddingMask bartInput
        attentionMask = ilift $ mkTransformerAttentionMask bartDataType bartAttentionMaskBias inputPaddingMask
        pos = flip addScalar (2 :: Int) <<$>> ilift (mkPos bartInput)
     in runIxStateT $
          ireturn bartDecoderInput
            >>>= IxStateT . forward bartShiftRightDecoderInput
            >>>= ( \rightShiftedDecoderInput ->
                     let decoderPos = flip addScalar (2 :: Int) <<$>> ilift (mkPos rightShiftedDecoderInput)
                         crossAttentionMask =
                           ilift $
                             mkTransformerCrossAttentionMask
                               bartDataType
                               (sShape rightShiftedDecoderInput)
                               bartAttentionMaskBias
                               inputPaddingMask
                      in ireturn (mkBARTPaddingMask bartDecoderInput)
                           >>>= IxStateT . forward bartShiftRightPaddingMask
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          ilift $
                                            mkTransformerDecoderAttentionMask
                                              bartDataType
                                              bartAttentionMaskBias
                                              rightShiftedDecoderInputPaddingMask
                                     in SequenceToSequenceTransformerInput
                                          <<$>> ireturn bartInput
                                          <<*>> ireturn rightShiftedDecoderInput
                                          <<*>> pos
                                          <<*>> decoderPos
                                          <<*>> attentionMask
                                          <<*>> decoderAttentionMask
                                          <<*>> crossAttentionMask
                                )
                           >>>= IxStateT . forward bartModel
                           >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ BARTOutput decoderOutput encoderOutput inputPaddingMask
                                )
                 )

testBart = do
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
      attentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: seqDim :|: seqDim :|: SNil)
      decoderInput = sOnes' (SDataType SInt64) (SShape $ batchDim :|: decoderSeqDim :|: SNil)
      decoderAttentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: decoderSeqDim :|: SNil)
      crossAttentionMask = sOnes' bartDataType (SShape $ SName @"*" :&: SSize @1 :|: decoderSeqDim :|: seqDim :|: SNil)
      (bartModel, g') = initialize @(SequenceToSequenceTransformer 'BART 'WithLMHead 4 4 _ _ _ _ _ _ _ _ _ _ _) (gradient, device, bartDataType, headDim, headEmbedDim, embedDim, inputEmbedDim, ffnDim, bartPosEncDim, vocabDim, bartDropoutP, bartEps) g
  (bartOutput, g'') <-
    let pos = sOnes' (SDataType SInt64) (SShape $ seqDim :|: SNil)
        decoderPos = sOnes' (SDataType SInt64) (SShape $ decoderSeqDim :|: SNil)
     in forward bartModel SequenceToSequenceTransformerInput {..} g'
  (bartOutput', g''') <-
    let bartShiftRightDecoderInput = ShiftRight bartEOSTokenId
        bartShiftRightPaddingMask = ShiftRight 0
        model = BARTModel (GBARTModel {..})
        inputs = BARTInput input decoderInput
     in forward model inputs g''
  pure ((bartOutput, bartOutput'), g''')

-- | 'HasForward' instance for BART models.
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
    MkTransformerAttentionMaskC BARTDataType inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim attentionMask,
    MkTransformerCrossAttentionMaskC BARTDataType rightShiftedDecoderInputShape rightShiftedDecoderInputSeqDim inputPaddingMaskGradient inputPaddingMaskLayout inputPaddingMaskDevice inputPaddingMaskDataType inputPaddingMaskShape inputPaddingMaskSeqDim crossAttentionMask,
    MkTransformerDecoderAttentionMaskC BARTDataType rightShiftedDecoderInputPaddingMaskLayout rightShiftedDecoderInputPaddingMaskDevice rightShiftedDecoderInputPaddingMaskShape rightShiftedDecoderInputPaddingMaskSeqDim decoderAttentionMask,
    HasForward (ShiftRight Int) decoderInput generator rightShiftedDecoderInput generator,
    HasForward (ShiftRight Int) decoderInputPaddingMask generator rightShiftedDecoderInputPaddingMask generator,
    HasForward
      (SequenceToSequenceTransformer 'BART transformerHead numLayers numLayers gradient device BARTDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim BARTPosEncDim vocabDim BARTDropoutP)
      (SequenceToSequenceTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generator
      (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
      generatorOutput
  ) =>
  HasForward
    (BARTModel transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
    (BARTGenerationInput decoderInput encoderOutput inputPaddingMask)
    generator
    (BARTOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutput
  where
  forward (BARTModel GBARTModel {..}) BARTGenerationInput {..} =
    runIxStateT $
      ireturn bartGenerationDecoderInput
        >>>= IxStateT . forward bartShiftRightDecoderInput
        >>>= ( \rightShiftedDecoderInput ->
                 let decoderPos = flip addScalar (2 :: Int) <<$>> ilift (mkPos rightShiftedDecoderInput)
                     crossAttentionMask =
                       ilift $
                         mkTransformerCrossAttentionMask
                           bartDataType
                           (sShape rightShiftedDecoderInput)
                           bartAttentionMaskBias
                           bartGenerationInputPaddingMask
                  in ireturn (mkBARTPaddingMask bartGenerationDecoderInput)
                       >>>= IxStateT . forward bartShiftRightPaddingMask
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      ilift $
                                        mkTransformerDecoderAttentionMask
                                          bartDataType
                                          bartAttentionMaskBias
                                          rightShiftedDecoderInputPaddingMask
                                 in SequenceToSequenceTransformerGenerationInput
                                      <<$>> ireturn rightShiftedDecoderInput
                                      <<*>> ireturn bartGenerationEncoderOutput
                                      <<*>> decoderPos
                                      <<*>> decoderAttentionMask
                                      <<*>> crossAttentionMask
                            )
                       >>>= IxStateT . forward bartModel
                       >>>= ( \(SequenceToSequenceTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ BARTOutput decoderOutput encoderOutput bartGenerationInputPaddingMask
                            )
             )
