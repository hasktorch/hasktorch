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
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.Pegasus.Common where

import Control.Monad.Catch (MonadThrow)
import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxStateT (..))
import Control.Monad.Indexed.Trans (IxMonadTrans (ilift))
import Data.Functor.Indexed ((<<$>>), (<<*>>))
import Data.Kind (Type)
import Data.Singletons (SingI (..))
import Data.Singletons.TypeLits (SNat)
import GHC.TypeLits (Nat, Symbol)
import Torch.GraduallyTyped.DType (DType (..), DataType (..), SDType (..), SDataType (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), SDevice)
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasStateDict (..), ModelSpec)
import Torch.GraduallyTyped.NN.Transformer.GEncoderDecoder (EDTDecoderF, EDTEncoderF, EDTHeadF, EDTSharedEmbeddingF, EncoderDecoderTransformerGenerationInput (..), EncoderDecoderTransformerInput (..), EncoderDecoderTransformerOutput (..), GEncoderDecoderTransformer, encoderDecoderTransformerSpec)
import Torch.GraduallyTyped.NN.Transformer.Type (MkPosC, MkTransformerAttentionMaskC, MkTransformerCrossAttentionMaskC, MkTransformerDecoderAttentionMaskC, MkTransformerPaddingMaskC, STransformerHead, STransformerStyle (SPegasus), ShiftRight (..), TransformerHead (..), TransformerStyle (Pegasus), mkPos, mkTransformerAttentionMask, mkTransformerCrossAttentionMask, mkTransformerDecoderAttentionMask, mkTransformerInput, mkTransformerPaddingMask)
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

-- | Pegasus dropout rate.
-- 'dropout_rate = 0.1'
pegasusDropoutP :: Double
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

type instance
  ModelSpec (GPegasusModel pegasusModel) =
    GPegasusModel (ModelSpec pegasusModel)

-- | Specifies the Pegasus model.
type family
  PegasusModelF
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
  PegasusModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim =
    GPegasusModel
      ( GEncoderDecoderTransformer
          inputEmbedDim
          (EDTEncoderF 'Pegasus numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim)
          (EDTDecoderF 'Pegasus numLayers gradient device PegasusDataType headDim headEmbedDim embedDim inputEmbedDim ffnDim PegasusPosEncDim)
          (EDTSharedEmbeddingF 'Pegasus gradient device PegasusDataType inputEmbedDim vocabDim)
          (EDTHeadF 'Pegasus transformerHead gradient device PegasusDataType inputEmbedDim vocabDim)
      )

-- | Specifies the parameters of a Pegasus model.
--
-- - @transformerHead@: the head of the Pegasus model.
-- - @numLayers@: the number of layers in the Pegasus model.
-- - @gradient@: whether to compute the gradient of the Pegasus model.
-- - @device@: the computational device on which the Pegasus model parameters are to be allocated.
pegasusModelSpec ::
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
  ModelSpec (PegasusModelF transformerHead numLayers gradient device headDim headEmbedDim embedDim inputEmbedDim ffnDim vocabDim)
pegasusModelSpec transformerHead numLayers gradient device =
  GPegasusModel
    ( encoderDecoderTransformerSpec
        SPegasus
        transformerHead
        numLayers
        numLayers
        gradient
        device
        pegasusDataType
        (sing @headDim)
        (sing @headEmbedDim)
        (sing @embedDim)
        (sing @inputEmbedDim)
        (sing @ffnDim)
        pegasusPosEncDim
        (sing @vocabDim)
        pegasusDropoutP
        pegasusEps
    )
    (ShiftRight pegasusBOSTokenId)
    (ShiftRight 0)

instance HasStateDict modelSpec => HasStateDict (GPegasusModel modelSpec) where
  fromStateDict (GPegasusModel modelSpec decoderInputShiftSpec paddingMaskShiftSpec) k =
    GPegasusModel
      <$> fromStateDict modelSpec k
      <*> fromStateDict decoderInputShiftSpec k
      <*> fromStateDict paddingMaskShiftSpec k
  toStateDict k GPegasusModel {..} = toStateDict k pegasusModel

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
-- The padding and attention masks are shifted to the right as well.
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
    HasForward (ShiftRight Int) decoderInput generatorDevice rightShiftedDecoderInput generatorDevice,
    HasForward (ShiftRight Int) decoderInputPaddingMask generatorDevice rightShiftedDecoderInputPaddingMask generatorDevice,
    HasForward
      pegasusModel
      (EncoderDecoderTransformerInput input rightShiftedDecoderInput pos decoderPos attentionMask decoderAttentionMask crossAttentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GPegasusModel pegasusModel)
    (PegasusInput input decoderInput)
    generatorDevice
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutputDevice
  where
  forward GPegasusModel {..} PegasusInput {..} =
    let inputPaddingMask = mkPegasusPaddingMask pegasusInput
        attentionMask = ilift $ mkTransformerAttentionMask pegasusDataType pegasusAttentionMaskBias inputPaddingMask
        pos = ilift $ mkPos pegasusInput
     in runIxStateT $
          ireturn pegasusDecoderInput
            >>>= IxStateT . forward pegasusShiftRightDecoderInput
            >>>= ( \rightShiftedDecoderInput ->
                     let decoderPos =
                           ilift $ mkPos rightShiftedDecoderInput
                         crossAttentionMask =
                           ilift $
                             mkTransformerCrossAttentionMask
                               pegasusDataType
                               (sShape rightShiftedDecoderInput)
                               pegasusAttentionMaskBias
                               inputPaddingMask
                      in ireturn (mkPegasusPaddingMask pegasusDecoderInput)
                           >>>= IxStateT . forward pegasusShiftRightPaddingMask
                           >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                    let decoderAttentionMask =
                                          ilift $
                                            mkTransformerDecoderAttentionMask
                                              pegasusDataType
                                              pegasusAttentionMaskBias
                                              rightShiftedDecoderInputPaddingMask
                                     in EncoderDecoderTransformerInput
                                          <<$>> ireturn pegasusInput
                                          <<*>> ireturn rightShiftedDecoderInput
                                          <<*>> pos
                                          <<*>> decoderPos
                                          <<*>> attentionMask
                                          <<*>> decoderAttentionMask
                                          <<*>> crossAttentionMask
                                )
                           >>>= IxStateT . forward pegasusModel
                           >>>= ( \(EncoderDecoderTransformerOutput decoderOutput encoderOutput) ->
                                    ireturn $ PegasusOutput decoderOutput encoderOutput inputPaddingMask
                                )
                 )

-- | 'HasForward' instance for Pegasus models.
-- Use this instance for sequence generation once the encoder's output is available.
--
-- Note that this instance always shifts decoder inputs to the right
-- by adding a BOS token at the beginning.
-- The padding and attention masks are shifted to the right as well.
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
    HasForward (ShiftRight Int) decoderInput generatorDevice rightShiftedDecoderInput generatorDevice,
    HasForward (ShiftRight Int) decoderInputPaddingMask generatorDevice rightShiftedDecoderInputPaddingMask generatorDevice,
    HasForward
      pegasusModel
      (EncoderDecoderTransformerGenerationInput rightShiftedDecoderInput encoderOutput decoderPos decoderAttentionMask crossAttentionMask)
      generatorDevice
      (EncoderDecoderTransformerOutput decoderOutput encoderOutput)
      generatorOutputDevice
  ) =>
  HasForward
    (GPegasusModel pegasusModel)
    (PegasusGenerationInput decoderInput encoderOutput inputPaddingMask)
    generatorDevice
    (PegasusOutput decoderOutput encoderOutput inputPaddingMask)
    generatorOutputDevice
  where
  forward GPegasusModel {..} PegasusGenerationInput {..} =
    runIxStateT $
      ireturn pegasusGenerationDecoderInput
        >>>= IxStateT . forward pegasusShiftRightDecoderInput
        >>>= ( \rightShiftedDecoderInput ->
                 let decoderPos =
                       ilift $ mkPos rightShiftedDecoderInput
                     crossAttentionMask =
                       ilift $
                         mkTransformerCrossAttentionMask
                           pegasusDataType
                           (sShape rightShiftedDecoderInput)
                           pegasusAttentionMaskBias
                           pegasusGenerationInputPaddingMask
                  in ireturn (mkPegasusPaddingMask pegasusGenerationDecoderInput)
                       >>>= IxStateT . forward pegasusShiftRightPaddingMask
                       >>>= ( \rightShiftedDecoderInputPaddingMask ->
                                let decoderAttentionMask =
                                      ilift $
                                        mkTransformerDecoderAttentionMask
                                          pegasusDataType
                                          pegasusAttentionMaskBias
                                          rightShiftedDecoderInputPaddingMask
                                 in EncoderDecoderTransformerGenerationInput
                                      <<$>> ireturn rightShiftedDecoderInput
                                      <<*>> ireturn pegasusGenerationEncoderOutput
                                      <<*>> decoderPos
                                      <<*>> decoderAttentionMask
                                      <<*>> crossAttentionMask
                            )
                       >>>= IxStateT . forward pegasusModel
                       >>>= ( \(EncoderDecoderTransformerOutput decoderOutput encoderOutput) ->
                                ireturn $ PegasusOutput decoderOutput encoderOutput pegasusGenerationInputPaddingMask
                            )
             )
