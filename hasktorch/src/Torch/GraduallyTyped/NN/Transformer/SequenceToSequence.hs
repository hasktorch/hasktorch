{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -v2 #-}

module Torch.GraduallyTyped.NN.Transformer.SequenceToSequence where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Constraint, Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearWithoutBiasC, Linear)
import Torch.GraduallyTyped.NN.Sparse (Embedding, HasInitializeEmbeddingC)
import Torch.GraduallyTyped.NN.Transformer.Decoder (HasInitializeTransformerDecoderC, TransformerDecoder)
import Torch.GraduallyTyped.NN.Transformer.Encoder (HasInitializeTransformerEncoderC, TransformerEncoder)
import Torch.GraduallyTyped.NN.Transformer.Type (TransformerStyle (..))
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Shape.Type (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar)
import Torch.GraduallyTyped.Tensor.Type (Tensor)

-- | Sequence-to-sequence transformer model.
data
  SequenceToSequenceTransformer
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformer ::
    forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP.
    { -- | encoder
      seqToSeqEncoder :: TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP,
      -- | decoder
      seqToSeqDecoder :: TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP,
      -- | shared embedding
      seqToSeqEmbedding :: Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing
    } ->
    SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

-- | Sequence-to-sequence transformer model with language modelling head.
data
  SequenceToSequenceTransformerWithLMHead
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (style :: TransformerStyle)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (posEncDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformerWithLMHead ::
    forall numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP.
    { -- | sequence-to-sequence transformer
      seqToSeqTransformer :: SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP,
      -- | language modelling head
      seqToSeqLMHead :: Linear 'WithoutBias device dataType inputEmbedDim vocabDim,
      -- | input embedding dim for scaling
      seqToSeqInputEmbedDim :: Dim String Integer
    } ->
    SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP

type HasInitializeSequenceToSequenceTransformerC
  (transformer :: Type)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (posEncDim :: Dim (Name Symbol) (Size Nat))
  (vocabDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device))))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device)))))),
    WithDimC inputEmbedDim (WithDimF ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device))))),
    WithDimC ffnDim (WithDimF posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device)))),
    WithDimC posEncDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device))),
    WithDimC vocabDim (dropoutP -> Double -> Generator device -> (transformer, Generator device))
  )

instance
  ( HasInitializeSequenceToSequenceTransformerC (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP,
    HasInitialize (TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP),
    HasInitializeTransformerEncoderC numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP,
    HasInitialize (TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP),
    HasInitializeTransformerDecoderC numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP,
    HasInitializeEmbeddingC ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing
  ) =>
  HasInitialize (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( WithDimF
                        embedDim
                        ( WithDimF
                            inputEmbedDim
                            ( WithDimF
                                ffnDim
                                ( WithDimF
                                    posEncDim
                                    ( WithDimF
                                        vocabDim
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @inputEmbedDim $
                          \inputEmbedDim ->
                            withDim @ffnDim $
                              \ffnDim ->
                                withDim @posEncDim $
                                  \posEncDim ->
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device)) $
                                      \vocabDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps = runState $ do
        encoder <-
          state $
            withoutDim @posEncDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
                                          )
                                          deviceType
                                      )
                                      dType
                                  )
                                  headDim
                              )
                              headEmbedDim
                          )
                          embedDim
                      )
                      inputEmbedDim
                  )
                  ffnDim
              )
              posEncDim
              dropoutP
              eps
        decoder <-
          state $
            withoutDim @posEncDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @inputEmbedDim
                          ( withoutDim @embedDim
                              ( withoutDim @headEmbedDim
                                  ( withoutDim @headDim
                                      ( withoutDataType @dataType
                                          ( withoutDevice @device
                                              ( initialize @(TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
                                              )
                                              deviceType
                                          )
                                          dType
                                      )
                                      headDim
                                  )
                                  headEmbedDim
                              )
                              embedDim
                          )
                          inputEmbedDim
                      )
                      inputEmbedDim
                  )
                  ffnDim
              )
              posEncDim
              dropoutP
              eps
        embedding <-
          state $
            withoutDim @inputEmbedDim
              ( withoutDim @vocabDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                          )
                          deviceType
                      )
                      dType
                  )
                  vocabDim
              )
              inputEmbedDim
        pure $ SequenceToSequenceTransformer encoder decoder embedding

instance
  ( HasInitializeSequenceToSequenceTransformerC (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP,
    HasInitializeSequenceToSequenceTransformerC (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP,
    HasInitialize (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP),
    HasInitializeLinearWithoutBiasC device dataType inputEmbedDim vocabDim
  ) =>
  HasInitialize (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                headDim
                ( WithDimF
                    headEmbedDim
                    ( WithDimF
                        embedDim
                        ( WithDimF
                            inputEmbedDim
                            ( WithDimF
                                ffnDim
                                ( WithDimF
                                    posEncDim
                                    ( WithDimF
                                        vocabDim
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
  initialize =
    withDevice @device $
      \deviceType ->
        withDataType @dataType $
          \dType ->
            withDim @headDim $
              \headDim ->
                withDim @headEmbedDim $
                  \headEmbedDim ->
                    withDim @embedDim $
                      \embedDim ->
                        withDim @inputEmbedDim $
                          \inputEmbedDim ->
                            withDim @ffnDim $
                              \ffnDim ->
                                withDim @posEncDim $
                                  \posEncDim ->
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP, Generator device)) $
                                      \vocabDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP eps = runState $ do
        transformer <-
          state $
            withoutDim @vocabDim
              ( withoutDim @posEncDim
                  ( withoutDim @ffnDim
                      ( withoutDim @inputEmbedDim
                          ( withoutDim @embedDim
                              ( withoutDim @headEmbedDim
                                  ( withoutDim @headDim
                                      ( withoutDataType @dataType
                                          ( withoutDevice @device
                                              ( initialize @(SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
                                              )
                                              deviceType
                                          )
                                          dType
                                      )
                                      headDim
                                  )
                                  headEmbedDim
                              )
                              embedDim
                          )
                          inputEmbedDim
                      )
                      ffnDim
                  )
                  posEncDim
              )
              vocabDim
              dropoutP
              eps
        lmHead <-
          state $
            withoutDim @vocabDim
              ( withoutDim @inputEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  inputEmbedDim
              )
              vocabDim
        pure $ SequenceToSequenceTransformerWithLMHead transformer lmHead inputEmbedDim

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for training.
data SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerInput ::
    forall input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask.
    { input :: input,
      decoderInput :: decoderInput,
      relPos :: relPos,
      decoderRelPos :: decoderRelPos,
      attentionMask :: attentionMask,
      decoderAttentionMask :: decoderAttentionMask,
      crossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask

deriving instance
  ( Show input,
    Show decoderInput,
    Show relPos,
    Show decoderRelPos,
    Show attentionMask,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)

-- | Output data type for use with a sequence-to-sequence transformer.
data SequenceToSequenceTransformerOutput decoderOutput encoderOutput where
  SequenceToSequenceTransformerOutput ::
    forall decoderOutput encoderOutput.
    { decoderOutput :: decoderOutput,
      encoderOutput :: encoderOutput
    } ->
    SequenceToSequenceTransformerOutput decoderOutput encoderOutput

deriving instance
  ( Show decoderOutput,
    Show encoderOutput
  ) =>
  Show (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)

-- | Input data type for use with a sequence-to-sequence transformer.
-- Use this for inference.
data SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask where
  SequenceToSequenceTransformerGenerationInput ::
    forall decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask.
    { generationDecoderInput :: decoderInput,
      generationEncoderOutput :: encoderOutput,
      generationDecoderRelPos :: decoderRelPos,
      generationDecoderAttentionMask :: decoderAttentionMask,
      generationCrossAttentionMask :: crossAttentionMask
    } ->
    SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask

deriving instance
  ( Show decoderInput,
    Show encoderOutput,
    Show decoderRelPos,
    Show decoderAttentionMask,
    Show crossAttentionMask
  ) =>
  Show (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)

-- | 'HasForward' instance for sequence-to-sequence transformers without language modelling head.
-- Use this instance for training.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    HasForward
      (TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim dropoutP)
      (embeddingOutput, relPos, attentionMask)
      embeddingGeneratorOutput
      encoderOutput
      encoderGeneratorOutput,
    HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      decoderInput
      encoderGeneratorOutput
      embeddingOutput'
      embeddingGeneratorOutput',
    HasForward
      (TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderRelPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward SequenceToSequenceTransformer {..} SequenceToSequenceTransformerInput {..} =
    runIxState $
      ireturn input
        >>>= IxState . forward seqToSeqEmbedding
        >>>= (\input' -> IxState $ forward seqToSeqEncoder (input', relPos, attentionMask))
        >>>= ( \encoderOutput ->
                 ireturn decoderInput
                   >>>= IxState . forward seqToSeqEmbedding
                   >>>= ( \decoderInput' ->
                            IxState $ forward seqToSeqDecoder (decoderInput', encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask)
                        )
                   >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
             )

-- | 'HasForward' instance for sequence-to-sequence transformers without language modelling head.
-- Use this instance for inference.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      decoderInput
      generator
      embeddingOutput'
      embeddingGeneratorOutput',
    HasForward
      (TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim posEncDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderRelPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward SequenceToSequenceTransformer {..} SequenceToSequenceTransformerGenerationInput {..} =
    runIxState $
      ireturn generationDecoderInput
        >>>= IxState . forward seqToSeqEmbedding
        >>>= ( \decoderInput' ->
                 IxState $ forward seqToSeqDecoder (decoderInput', generationEncoderOutput, generationDecoderRelPos, generationDecoderAttentionMask, generationCrossAttentionMask)
             )
        >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput generationEncoderOutput)

-- | 'HasForward' instance for sequence-to-sequence transformers with language modelling head.
instance
  ( HasForward
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
      input
      generator
      seqToSeqOutput
      seqToSeqGeneratorOutput,
    seqToSeqOutput ~ SequenceToSequenceTransformerOutput decoderOutput encoderOutput,
    HasForward
      (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
      decoderOutput
      seqToSeqGeneratorOutput
      (Tensor requiresGradient' layout' device' dataType' shape)
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformerWithLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim posEncDim vocabDim dropoutP)
    input
    generator
    (SequenceToSequenceTransformerOutput (Tensor requiresGradient' layout' device' dataType' shape) encoderOutput)
    generatorOutput
  where
  forward SequenceToSequenceTransformerWithLMHead {..} input =
    let scaling :: Double = sqrt . fromIntegral . dimSize $ seqToSeqInputEmbedDim
     in runIxState $
          ireturn input
            >>>= IxState . forward seqToSeqTransformer
            >>>= ( \SequenceToSequenceTransformerOutput {..} ->
                     ireturn decoderOutput
                       >>>= IxState . forward seqToSeqLMHead
                       >>>= ireturn . flip divScalar scaling
                       >>>= \decoderOutput' -> ireturn (SequenceToSequenceTransformerOutput decoderOutput' encoderOutput)
                 )

testForwardSeqToSeq :: _
testForwardSeqToSeq =
  let seqToSeq =
        undefined ::
          SequenceToSequenceTransformerWithLMHead
            128
            128
            'T5
            ('Device 'CPU)
            ('DataType 'Float)
            ('Dim ('Name "*") ('Size 8)) -- headDim
            ('Dim ('Name "*") ('Size 64)) -- headEmbedDim
            ('Dim ('Name "*") ('Size 512)) -- embedDim
            ('Dim ('Name "*") ('Size 512)) -- inputEmbedDim
            ('Dim ('Name "*") ('Size 2048)) -- ffnDim
            ('Dim ('Name "*") ('Size 32)) -- posEncDim
            ('Dim ('Name "*") ('Size 32128)) -- vocabDim
            Float
      input =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7)])
      decoderInput =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5)])
      relPos =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
      decoderRelPos =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Int64)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
      attentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
      decoderAttentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
      crossAttentionMask =
        undefined ::
          Tensor
            'WithoutGradient
            ('Layout 'Dense)
            ('Device 'CPU)
            ('DataType 'Float)
            ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 7)])
      g = undefined :: Generator ('Device 'CPU)
   in forward seqToSeq (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask) g
