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

-- | Whether or not the transformer comes with a language modelling head.
data HasLMHead = WithLMHead | WithoutLMHead

-- | Sequence-to-sequence transformer model.
data
  SequenceToSequenceTransformer
    (hasLMHead :: HasLMHead)
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
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  -- | T5-style sequence-to-sequence transformer without language modelling head.
  T5WithoutLMHead ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP.
    { -- | encoder
      seqToSeqWithoutLMHeadEncoder :: TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | decoder
      seqToSeqWithoutLMHeadDecoder :: TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | embedding
      seqToSeqWithoutLMHeadEmbedding :: Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing
    } ->
    SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP
  -- | T5-style sequence-to-sequence transformer with language modelling head.
  T5WithLMHead ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP.
    { -- | encoder
      seqToSeqWithLMHeadEncoder :: TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | decoder
      seqToSeqWithLMHeadDecoder :: TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | embedding
      seqToSeqWithLMHeadEmbedding :: Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
      -- | language model head
      seqToSeqLMHead :: Linear 'WithoutBias device dataType inputEmbedDim vocabDim,
      -- | input embed dim
      seqToSeqInputEmbedDim :: Dim String Integer
    } ->
    SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP

type family
  HasInitializeSequenceToSequenceTransformerC'
    (hasLMHead :: HasLMHead)
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
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (vocabDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type) ::
    Constraint
  where
  HasInitializeSequenceToSequenceTransformerC' 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP =
    HasInitializeEmbeddingC ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing
  HasInitializeSequenceToSequenceTransformerC' 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP =
    ( HasInitializeEmbeddingC ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
      HasInitializeLinearWithoutBiasC device dataType inputEmbedDim vocabDim
    )

type HasInitializeSequenceToSequenceTransformerC
  (hasLMHead :: HasLMHead)
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
  (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
  (vocabDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type) =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))),
    WithDimC inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))),
    WithDimC ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))),
    WithDimC relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))),
    WithDimC vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)),
    HasInitializeSequenceToSequenceTransformerC' hasLMHead numEncoderLayers numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP,
    HasInitialize (TransformerEncoder numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP),
    HasInitializeTransformerEncoderC numEncoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
    HasInitialize (TransformerDecoder numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP),
    HasInitializeTransformerDecoderC numDecoderLayers style device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP
  )

instance
  HasInitializeSequenceToSequenceTransformerC 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP =>
  HasInitialize (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) =
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
                                    relPosEncBucketDim
                                    ( WithDimF
                                        vocabDim
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))
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
                                withDim @relPosEncBucketDim $
                                  \relPosEncBucketDim ->
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)) $
                                      \vocabDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP eps = runState $ do
        encoder <-
          state $
            withoutDim @relPosEncBucketDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
              relPosEncBucketDim
              dropoutP
              eps
        decoder <-
          state $
            withoutDim @relPosEncBucketDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @inputEmbedDim
                          ( withoutDim @embedDim
                              ( withoutDim @headEmbedDim
                                  ( withoutDim @headDim
                                      ( withoutDataType @dataType
                                          ( withoutDevice @device
                                              ( initialize @(TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
              relPosEncBucketDim
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
        pure $ T5WithoutLMHead encoder decoder embedding

instance
  ( HasInitializeSequenceToSequenceTransformerC 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP
  ) =>
  HasInitialize (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) =
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
                                    relPosEncBucketDim
                                    ( WithDimF
                                        vocabDim
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))
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
                                withDim @relPosEncBucketDim $
                                  \relPosEncBucketDim ->
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)) $
                                      \vocabDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP eps = runState $ do
        encoder <-
          state $
            withoutDim @relPosEncBucketDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
              relPosEncBucketDim
              dropoutP
              eps
        decoder <-
          state $
            withoutDim @relPosEncBucketDim
              ( withoutDim @ffnDim
                  ( withoutDim @inputEmbedDim
                      ( withoutDim @inputEmbedDim
                          ( withoutDim @embedDim
                              ( withoutDim @headEmbedDim
                                  ( withoutDim @headDim
                                      ( withoutDataType @dataType
                                          ( withoutDevice @device
                                              ( initialize @(TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
              relPosEncBucketDim
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
        pure $ T5WithLMHead encoder decoder embedding lmHead inputEmbedDim

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

-- | 'HasForward' instance for T5-style sequence-to-sequence transformers without language modelling head.
-- Use this instance for training.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    HasForward
      (TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
      (TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
    (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward T5WithoutLMHead {..} SequenceToSequenceTransformerInput {..} =
    runIxState $
      ireturn input
        >>>= IxState . forward seqToSeqWithoutLMHeadEmbedding
        >>>= (\input' -> IxState $ forward seqToSeqWithoutLMHeadEncoder (input', relPos, attentionMask))
        >>>= ( \encoderOutput ->
                 ireturn decoderInput
                   >>>= IxState . forward seqToSeqWithoutLMHeadEmbedding
                   >>>= ( \decoderInput' ->
                            IxState $ forward seqToSeqWithoutLMHeadDecoder (decoderInput', encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask)
                        )
                   >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
             )

-- | 'HasForward' instance for T5-style sequence-to-sequence transformers without language modelling head.
-- Use this instance for inference.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      decoderInput
      generator
      embeddingOutput'
      embeddingGeneratorOutput',
    HasForward
      (TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
    (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
    generatorOutput
  where
  forward T5WithoutLMHead {..} SequenceToSequenceTransformerGenerationInput {..} =
    runIxState $
      ireturn generationDecoderInput
        >>>= IxState . forward seqToSeqWithoutLMHeadEmbedding
        >>>= ( \decoderInput' ->
                 IxState $ forward seqToSeqWithoutLMHeadDecoder (decoderInput', generationEncoderOutput, generationDecoderRelPos, generationDecoderAttentionMask, generationCrossAttentionMask)
             )
        >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput generationEncoderOutput)

-- | 'HasForward' instance for T5-style sequence-to-sequence transformers with language modelling head.
-- Use this instance for training.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      input
      generator
      embeddingOutput
      embeddingGeneratorOutput,
    HasForward
      (TransformerEncoder numEncoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
      (TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderRelPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      decoderGeneratorOutput,
    HasForward
      (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
      decoderOutput
      decoderGeneratorOutput
      (Tensor requiresGradient' layout' device' dataType' shape)
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput (Tensor requiresGradient' layout' device' dataType' shape) encoderOutput)
    generatorOutput
  where
  forward T5WithLMHead {..} SequenceToSequenceTransformerInput {..} =
    let scaling :: Double = sqrt . fromIntegral . dimSize $ seqToSeqInputEmbedDim
     in runIxState $
          ireturn input
            >>>= IxState . forward seqToSeqWithLMHeadEmbedding
            >>>= (\input' -> IxState $ forward seqToSeqWithLMHeadEncoder (input', relPos, attentionMask))
            >>>= ( \encoderOutput ->
                     ireturn decoderInput
                       >>>= IxState . forward seqToSeqWithLMHeadEmbedding
                       >>>= ( \decoderInput' ->
                                IxState $ forward seqToSeqWithLMHeadDecoder (decoderInput', encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask)
                            )
                       >>>= IxState . forward seqToSeqLMHead
                       >>>= ireturn . flip divScalar scaling
                       >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput encoderOutput)
                 )

-- | 'HasForward' instance for T5-style sequence-to-sequence transformers with language modelling head.
-- Use this instance for inference.
instance
  ( HasForward
      (Embedding ('Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      decoderInput
      generator
      embeddingOutput'
      embeddingGeneratorOutput',
    HasForward
      (TransformerDecoder numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( embeddingOutput',
        encoderOutput,
        decoderRelPos,
        decoderAttentionMask,
        crossAttentionMask
      )
      embeddingGeneratorOutput'
      decoderOutput
      decoderGeneratorOutput,
    HasForward
      (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
      decoderOutput
      decoderGeneratorOutput
      (Tensor requiresGradient' layout' device' dataType' shape)
      generatorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers 'T5 device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    (SequenceToSequenceTransformerGenerationInput decoderInput encoderOutput decoderRelPos decoderAttentionMask crossAttentionMask)
    generator
    (SequenceToSequenceTransformerOutput (Tensor requiresGradient' layout' device' dataType' shape) encoderOutput)
    generatorOutput
  where
  forward T5WithLMHead {..} SequenceToSequenceTransformerGenerationInput {..} =
    let scaling :: Double = sqrt . fromIntegral . dimSize $ seqToSeqInputEmbedDim
     in runIxState $
          ireturn generationDecoderInput
            >>>= IxState . forward seqToSeqWithLMHeadEmbedding
            >>>= ( \decoderInput' ->
                     IxState $ forward seqToSeqWithLMHeadDecoder (decoderInput', generationEncoderOutput, generationDecoderRelPos, generationDecoderAttentionMask, generationCrossAttentionMask)
                 )
            >>>= IxState . forward seqToSeqLMHead
            >>>= ireturn . flip divScalar scaling
            >>>= \decoderOutput -> ireturn (SequenceToSequenceTransformerOutput decoderOutput generationEncoderOutput)

-- testForwardSeqToSeq :: _
-- testForwardSeqToSeq =
--   let seqToSeq =
--         undefined ::
--           SequenceToSequenceTransformer
--             'WithLMHead
--             128
--             128
--             'T5
--             ('Device 'CPU)
--             ('DataType 'Float)
--             ('Dim ('Name "*") ('Size 8)) -- headDim
--             ('Dim ('Name "*") ('Size 64)) -- headEmbedDim
--             ('Dim ('Name "*") ('Size 512)) -- embedDim
--             ('Dim ('Name "*") ('Size 512)) -- inputEmbedDim
--             ('Dim ('Name "*") ('Size 2048)) -- ffnDim
--             ('Dim ('Name "*") ('Size 32)) -- relPosEncBucketDim
--             ('Dim ('Name "*") ('Size 32128)) -- vocabDim
--             Float
--       input =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Int64)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7)])
--       decoderInput =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Int64)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5)])
--       relPos =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Int64)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
--       decoderRelPos =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Int64)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
--       attentionMask =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Float)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 7), 'Dim ('Name "*") ('Size 7)])
--       decoderAttentionMask =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Float)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 5)])
--       crossAttentionMask =
--         undefined ::
--           Tensor
--             'WithoutGradient
--             ('Layout 'Dense)
--             ('Device 'CPU)
--             ('DataType 'Float)
--             ('Shape '[ 'Dim ('Name "*") ('Size 1), 'Dim ('Name "*") ('Size 5), 'Dim ('Name "*") ('Size 7)])
--       g = undefined :: Generator ('Device 'CPU)
--    in forward seqToSeq (SequenceToSequenceTransformerInput input decoderInput relPos decoderRelPos attentionMask decoderAttentionMask crossAttentionMask) g
