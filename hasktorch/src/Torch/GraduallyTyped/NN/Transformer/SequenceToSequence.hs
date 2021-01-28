{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Torch.GraduallyTyped.NN.Transformer.SequenceToSequence where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearWithoutBiasC, Linear)
import Torch.GraduallyTyped.NN.Sparse (Embedding, HasInitializeEmbeddingC)
import Torch.GraduallyTyped.NN.Transformer.Decoder (HasInitializeTransformerDecoderC, TransformerDecoder)
import Torch.GraduallyTyped.NN.Transformer.Encoder (HasInitializeTransformerEncoderC, TransformerEncoder)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data HasLMHead = WithLMHead | WithoutLMHead

data
  SequenceToSequenceTransformer
    (hasLMHead :: HasLMHead)
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
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
  SequenceToSequenceTransformerWithoutLMHead ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP.
    { -- | encoder
      seqToSeqWithoutLMHeadEncoder :: TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | decoder
      seqToSeqWithoutLMHeadDecoder :: TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | embedding
      seqToSeqWithoutLMHeadEmbedding :: Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing
    } ->
    SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP
  SequenceToSequenceTransformerWithLMHead ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP.
    { -- | encoder
      seqToSeqWithLMHeadEncoder :: TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | decoder
      seqToSeqWithLMHeadDecoder :: TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | embedding
      seqToSeqWithLMHeadEmbedding :: Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
      -- | language model head
      seqToSeqLMHead :: Linear 'WithoutBias device dataType inputEmbedDim vocabDim
    } ->
    SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP

type HasInitializeSequenceToSequenceTransformerC hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP =
  ( HasInitializeTransformerEncoderC numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
    HasInitializeTransformerDecoderC numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
    HasInitializeEmbeddingC ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))))),
    WithDimC inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))))),
    WithDimC ffnDim (WithDimF relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)))),
    WithDimC relPosEncBucketDim (WithDimF vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))),
    WithDimC vocabDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer hasLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))
  )

instance
  HasInitializeSequenceToSequenceTransformerC 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP =>
  HasInitialize (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) =
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
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))
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
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)) $
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
                                          ( initialize @(TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
                                              ( initialize @(TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
                          ( initialize @(Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                          )
                          deviceType
                      )
                      dType
                  )
                  vocabDim
              )
              inputEmbedDim
        pure $ SequenceToSequenceTransformerWithoutLMHead encoder decoder embedding

instance
  ( HasInitializeSequenceToSequenceTransformerC 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP,
    HasInitializeLinearWithoutBiasC device dataType inputEmbedDim vocabDim
  ) =>
  HasInitialize (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP) =
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
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device))
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
                                    withDim @vocabDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP, Generator device)) $
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
                                          ( initialize @(TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
                                              ( initialize @(TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
                          ( initialize @(Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
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
        pure $ SequenceToSequenceTransformerWithLMHead encoder decoder embedding lmHead

instance
  ( HasForward
      (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
      (Generator generatorDevice),
    inputEmbeddingOutput
      ~ ForwardOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
          (Generator generatorDevice),
    inputEmbeddingGeneratorOutput
      ~ ForwardGeneratorOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
          (Generator generatorDevice),
    HasForward
      (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( inputEmbeddingOutput,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      inputEmbeddingGeneratorOutput,
    encoderOutput
      ~ ForwardOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( inputEmbeddingOutput,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          inputEmbeddingGeneratorOutput,
    encoderGeneratorOutput
      ~ ForwardGeneratorOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( inputEmbeddingOutput,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          inputEmbeddingGeneratorOutput,
    HasForward
      (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
      encoderGeneratorOutput,
    decoderInputEmbeddingOutput
      ~ ForwardOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
          encoderGeneratorOutput,
    decoderInputEmbeddingGeneratorOutput
      ~ ForwardGeneratorOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
          encoderGeneratorOutput,
    HasForward
      (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( decoderInputEmbeddingOutput,
        encoderOutput,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      decoderInputEmbeddingGeneratorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
        ( ForwardOutput
            (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
            (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
            ( ForwardGeneratorOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                )
            ),
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                (Generator generatorDevice),
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                (Generator generatorDevice)
            ),
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
            (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
            ( ForwardGeneratorOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                )
            )
        )
  type
    ForwardGeneratorOutput
      (SequenceToSequenceTransformer 'WithoutLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
        ( ForwardOutput
            (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
            (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
            ( ForwardGeneratorOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                )
            ),
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                (Generator generatorDevice),
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                (Generator generatorDevice)
            ),
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
            (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
            ( ForwardGeneratorOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                )
            )
        )
  forward SequenceToSequenceTransformerWithoutLMHead {..} (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask) =
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
             )

instance
  ( HasForward
      (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
      (Generator generatorDevice),
    inputEmbeddingOutput
      ~ ForwardOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
          (Generator generatorDevice),
    inputEmbeddingGeneratorOutput
      ~ ForwardGeneratorOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
          (Generator generatorDevice),
    HasForward
      (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( inputEmbeddingOutput,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      inputEmbeddingGeneratorOutput,
    encoderOutput
      ~ ForwardOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( inputEmbeddingOutput,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          inputEmbeddingGeneratorOutput,
    encoderGeneratorOutput
      ~ ForwardGeneratorOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( inputEmbeddingOutput,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          inputEmbeddingGeneratorOutput,
    HasForward
      (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
      (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
      encoderGeneratorOutput,
    decoderInputEmbeddingOutput
      ~ ForwardOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
          encoderGeneratorOutput,
    decoderInputEmbeddingGeneratorOutput
      ~ ForwardGeneratorOutput
          (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
          (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
          encoderGeneratorOutput,
    HasForward
      (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( decoderInputEmbeddingOutput,
        encoderOutput,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      decoderInputEmbeddingGeneratorOutput,
    decoderOutput
      ~ ForwardOutput
          (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( decoderInputEmbeddingOutput,
            encoderOutput,
            Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
            Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
            Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
          )
          decoderInputEmbeddingGeneratorOutput,
    decoderGeneratorOutput
      ~ ForwardGeneratorOutput
          (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( decoderInputEmbeddingOutput,
            encoderOutput,
            Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
            Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
            Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
          )
          decoderInputEmbeddingGeneratorOutput,
    HasForward
      (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
      decoderOutput
      decoderGeneratorOutput
  ) =>
  HasForward
    (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
        ( ForwardOutput
            (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                ),
              ForwardOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                ),
              Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
              Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
              Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                )
            )
        )
        ( ForwardGeneratorOutput
            (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                ),
              ForwardOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                ),
              Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
              Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
              Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                )
            )
        )
  type
    ForwardGeneratorOutput
      (SequenceToSequenceTransformer 'WithLMHead numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim vocabDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (Linear 'WithoutBias device dataType inputEmbedDim vocabDim)
        ( ForwardOutput
            (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                ),
              ForwardOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                ),
              Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
              Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
              Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                )
            )
        )
        ( ForwardGeneratorOutput
            (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( ForwardOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                ),
              ForwardOutput
                (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                ( ForwardOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice),
                  Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                  Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                )
                ( ForwardGeneratorOutput
                    (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                    (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                    (Generator generatorDevice)
                ),
              Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
              Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
              Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
            )
            ( ForwardGeneratorOutput
                (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape)
                ( ForwardGeneratorOutput
                    (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
                    ( ForwardOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice),
                      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
                      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
                    )
                    ( ForwardGeneratorOutput
                        (Embedding ( 'Layout 'Dense) device dataType vocabDim inputEmbedDim 'Nothing)
                        (Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape)
                        (Generator generatorDevice)
                    )
                )
            )
        )
  forward SequenceToSequenceTransformerWithLMHead {..} (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask) =
    runIxState $
      ireturn input
        >>>= IxState . forward seqToSeqWithLMHeadEmbedding
        >>>= (\input' -> IxState $ forward seqToSeqWithLMHeadEncoder (input', relPos, attentionMask))
        >>>= ( \encoderOutput ->
                 ireturn decoderInput
                   >>>= IxState . forward seqToSeqWithLMHeadEmbedding
                   >>>= ( \decoderInput' ->
                            IxState $ forward seqToSeqWithLMHeadDecoder (decoderInput', encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask)
                        )
             )
        >>>= IxState . forward seqToSeqLMHead