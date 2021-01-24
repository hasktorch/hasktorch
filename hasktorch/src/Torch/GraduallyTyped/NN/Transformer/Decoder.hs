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

module Torch.GraduallyTyped.NN.Transformer.Decoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout, LayoutType)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm)
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasForwardTransformerDecoderStack, HasForwardTransformerDecoderStackGeneratorOutput, HasForwardTransformerDecoderStackOutput, HasInitializeTransformerDecoderStack, HasInitializeTransformerDecoderStackC, TransformerDecoderStack)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  TransformerDecoder
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoder ::
    forall numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP.
    { -- | decoder layer stack
      tdStack :: TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
      -- | decoder layer norm
      tdLayerNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]),
      -- | decoder dropout
      tdDropout :: Dropout dropoutP
    } ->
    TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP

type HasInitializeTransformerDecoderC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP =
  ( HasInitializeTransformerDecoderStackC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderStack
      (1 <=? numLayers)
      numLayers
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      decoderInputEmbedDim
      encoderOutputEmbedDim
      ffnDim
      dropoutP,
    HasInitializeLayerNormWithoutBiasC device dataType ( 'Shape '[decoderInputEmbedDim]),
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC encoderOutputEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerDecoderC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP) =
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
                            decoderInputEmbedDim
                            ( WithDimF
                                encoderOutputEmbedDim
                                ( WithDimF
                                    ffnDim
                                    (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device))
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
                        withDim @decoderInputEmbedDim $
                          \decoderInputEmbedDim ->
                            withDim @encoderOutputEmbedDim $
                              \encoderOutputEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP, Generator device)) $
                                  \ffnDim -> go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP eps = runState $ do
        decoderStack <-
          state $
            withoutDim @ffnDim
              ( withoutDim @encoderOutputEmbedDim
                  ( withoutDim @decoderInputEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
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
                  encoderOutputEmbedDim
              )
              ffnDim
              dropoutP
              eps
        let layerNorm =
              withoutShape @( 'Shape '[decoderInputEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [decoderInputEmbedDim]
                eps
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ TransformerDecoder decoderStack layerNorm dropout

type HasForwardTransformerDecoderC
  (numLayers :: Nat)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (decoderInputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (encoderOutputEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type)
  (decoderInputRequiresGradient :: RequiresGradient)
  (decoderInputLayout :: Layout LayoutType)
  (decoderInputDevice :: Device (DeviceType Nat))
  (decoderInputDataType :: DataType DType)
  (decoderInputShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (encoderOutputRequiresGradient :: RequiresGradient)
  (encoderOutputLayout :: Layout LayoutType)
  (encoderOutputDevice :: Device (DeviceType Nat))
  (encoderOutputDataType :: DataType DType)
  (encoderOutputShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (decoderAttentionMaskRequiresGradient :: RequiresGradient)
  (decoderAttentionMaskLayout :: Layout LayoutType)
  (decoderAttentionMaskDevice :: Device (DeviceType Nat))
  (decoderAttentionMaskDataType :: DataType DType)
  (decoderAttentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (crossAttentionMaskRequiresGradient :: RequiresGradient)
  (crossAttentionMaskLayout :: Layout LayoutType)
  (crossAttentionMaskDevice :: Device (DeviceType Nat))
  (crossAttentionMaskDataType :: DataType DType)
  (crossAttentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (generatorDevice :: Device (DeviceType Nat)) =
  ( Scalar dropoutP,
    HasForwardTransformerDecoderStack (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice),
    HasForward
      (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
      (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
      (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice)),
    HasForward
      (Dropout dropoutP)
      ( ForwardOutput
          (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
          (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
          (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
      )
      ( ForwardGeneratorOutput
          (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
          (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
          (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
      )
  )

instance
  HasForwardTransformerDecoderC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice =>
  HasForward (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP) (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape, Tensor encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape, Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice)
  where
  type
    ForwardOutput (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP) (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape, Tensor encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape, Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice) =
      ForwardOutput
        (Dropout dropoutP)
        ( ForwardOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
            (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
            (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
        )
        ( ForwardGeneratorOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
            (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
            (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
        )
  type
    ForwardGeneratorOutput (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP) (Tensor decoderInputRequiresGradient decoderInputLayout decoderInputDevice decoderInputDataType decoderInputShape, Tensor encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape, Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice) =
      ForwardGeneratorOutput
        (Dropout dropoutP)
        ( ForwardOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
            (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
            (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
        )
        ( ForwardGeneratorOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]))
            (HasForwardTransformerDecoderStackOutput (1 <=? numLayers) 'False numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim decoderInputRequiresGradient decoderInputLayout (decoderInputDevice <+> generatorDevice) decoderInputDataType decoderInputShape encoderOutputRequiresGradient encoderOutputLayout encoderOutputDevice encoderOutputDataType encoderOutputShape decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape (decoderInputDevice <+> generatorDevice))
            (HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) 'False numLayers device (decoderInputDevice <+> generatorDevice) encoderOutputDevice decoderAttentionMaskDevice crossAttentionMaskDevice (decoderInputDevice <+> generatorDevice))
        )
  forward TransformerDecoder {..} (decoderInput, encoderOutput, decoderAttentionMask, crossAttentionMask) =
    runIxState $
      ireturn decoderInput
        >>>= IxState . forward tdDropout
        >>>= (\decoderInput' -> IxState $ forward tdStack (decoderInput', encoderOutput, decoderAttentionMask, crossAttentionMask))
        >>>= IxState . forward tdLayerNorm
        >>>= IxState . forward tdDropout