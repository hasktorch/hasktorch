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
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  SequenceToSequenceTransformer ::
    forall numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP.
    { -- | encoder
      seqToSeqEncoder :: TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
      -- | decoder
      seqToSeqDecoder :: TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP
    } ->
    SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP

type HasInitializeSequenceToSequenceTransformerC numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP =
  ( HasInitializeTransformerEncoderC numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
    HasInitializeTransformerDecoderC numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))),
    WithDimC inputEmbedDim (WithDimF decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))),
    WithDimC decoderInputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))),
    WithDimC relPosEncBucketDim (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
  )

instance
  HasInitializeSequenceToSequenceTransformerC numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP =>
  HasInitialize (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP)
  where
  type
    InitializeF (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP) =
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
                                    ( WithDimF
                                        relPosEncBucketDim
                                        (dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
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
                            withDim @decoderInputEmbedDim $
                              \decoderInputEmbedDim ->
                                withDim @ffnDim $
                                  \ffnDim ->
                                    withDim @relPosEncBucketDim @(dropoutP -> Double -> Generator device -> (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)) $
                                      \relPosEncBucketDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP eps = runState $ do
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
                      ( withoutDim @decoderInputEmbedDim
                          ( withoutDim @embedDim
                              ( withoutDim @headEmbedDim
                                  ( withoutDim @headDim
                                      ( withoutDataType @dataType
                                          ( withoutDevice @device
                                              ( initialize @(TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
              )
              relPosEncBucketDim
              dropoutP
              eps
        pure $ SequenceToSequenceTransformer encoder decoder

instance
  ( HasForward
      (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice),
    HasForward
      (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
      ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
        ForwardOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          (Generator generatorDevice),
        Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
        Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      ( ForwardGeneratorOutput
          (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
          ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
            Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
            Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
          )
          (Generator generatorDevice)
      )
  ) =>
  HasForward
    (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
        ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice),
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice)
        )
  type
    ForwardGeneratorOutput
      (SequenceToSequenceTransformer numEncoderLayers numDecoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim decoderInputEmbedDim ffnDim relPosEncBucketDim dropoutP)
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
        (TransformerDecoder numDecoderLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
        ( Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape,
          ForwardOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice),
          Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
          Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
          Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
        )
        ( ForwardGeneratorOutput
            (TransformerEncoder numEncoderLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
            ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
              Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
              Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
            )
            (Generator generatorDevice)
        )
  forward SequenceToSequenceTransformer {..} (input, decoderInput, relPos, decoderRelPos, attentionMask, decoderAttentionMask, crossAttentionMask) =
    runIxState $
      IxState (forward seqToSeqEncoder (input, relPos, attentionMask))
        >>>= (\encoderOutput -> IxState $ forward seqToSeqDecoder (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask))