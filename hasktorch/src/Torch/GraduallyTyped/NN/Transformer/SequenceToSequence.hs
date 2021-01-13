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

import Control.Monad.Indexed ((>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.Decoder (HasInitializeTransformerDecoderC, TransformerDecoder)
import Torch.GraduallyTyped.NN.Transformer.Encoder (HasInitializeTransformerEncoderC, TransformerEncoder)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)

data
  SequenceToSequenceTransformer
    (numEncoderLayers :: Nat)
    (numDecoderLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformer ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP.
    { -- | encoder
      seqToSeqEncoder :: TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
      -- | decoder
      seqToSeqDecoder :: TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP
    } ->
    SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP

type HasInitializeSequenceToSequenceTransformerC numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP =
  ( HasInitializeTransformerEncoderC numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderC numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC decoderInputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeSequenceToSequenceTransformerC numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP =>
  HasInitialize (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP) =
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
                                decoderInputEmbedDim
                                ( WithDimF
                                    ffnDim
                                    (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device))
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
                            withDim @decoderInputEmbedDim $
                              \decoderInputEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP, Generator device)) $
                                  \ffnDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP eps = runState $ do
        encoder <-
          state $
            withoutDim @ffnDim
              ( withoutDim @inputEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
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
              dropoutP
              eps
        decoder <-
          state $
            withoutDim @ffnDim
              ( withoutDim @inputEmbedDim
                  ( withoutDim @decoderInputEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP)
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
                      decoderInputEmbedDim
                  )
                  inputEmbedDim
              )
              ffnDim
              dropoutP
              eps
        pure $ SequenceToSequenceTransformer encoder decoder

instance
  ( HasForward
      (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice),
    HasForward
      (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        ForwardOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
          ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
            Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          (Generator generatorDevice),
        Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      ( ForwardGeneratorOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
          ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
            Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          (Generator generatorDevice)
      )
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP)
    ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
      Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
      Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP)
        ( Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
            ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice),
          Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
            ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice)
        )
  type
    ForwardGeneratorOutput
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape,
        Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim dropoutP)
        ( Tensor requiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
            ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice),
          Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
            ( Tensor requiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice)
        )
  forward SequenceToSequenceTransformer {..} (input, decoderInput, attentionMask, decoderAttentionMask, crossAttentionMask) =
    runIxState $
      IxState (forward seqToSeqEncoder (input, attentionMask))
        >>>= (\encoderOutput -> IxState $ forward seqToSeqDecoder (decoderInput, encoderOutput, decoderAttentionMask, crossAttentionMask))