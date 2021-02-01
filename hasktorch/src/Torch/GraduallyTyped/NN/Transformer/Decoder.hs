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
{-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL #-}

module Torch.GraduallyTyped.NN.Transformer.Decoder where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol, type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (..), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (..), LayoutType (..))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Sparse (EmbeddingF)
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm)
import Torch.GraduallyTyped.NN.Sparse (Embedding, HasInitializeEmbeddingC)
import Torch.GraduallyTyped.NN.Transformer.DecoderStack (HasInitializeTransformerDecoderStack, HasInitializeTransformerDecoderStackC, TransformerDecoderStack)
import Torch.GraduallyTyped.NN.Type (HasBias (..))
import Torch.GraduallyTyped.Prelude (Seq)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (..))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..))
import Torch.GraduallyTyped.Shape.Class (BroadcastShapesF)
import Torch.GraduallyTyped.Shape.Type (By (..), SelectDim (..))
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (TransposeF, UnsqueezeF, transpose, unsqueeze)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add)
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
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoder ::
    forall numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP.
    { -- | decoder layer stack
      tdStack :: TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP,
      -- | decoder layer norm
      tdLayerNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[decoderInputEmbedDim]),
      -- | decoder dropout
      tdDropout :: Dropout dropoutP,
      -- | relative positional encoding
      tdRelPosEnc :: Embedding ( 'Layout 'Dense) device dataType relPosEncBucketDim headDim 'Nothing
    } ->
    TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP

type HasInitializeTransformerDecoderC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP =
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
    HasInitializeEmbeddingC ( 'Layout 'Dense) device dataType relPosEncBucketDim headDim 'Nothing,
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))),
    WithDimC decoderInputEmbedDim (WithDimF encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))),
    WithDimC encoderOutputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))),
    WithDimC relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerDecoderC numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP =>
  HasInitialize (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP)
  where
  type
    InitializeF (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP) =
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
                                    ( WithDimF
                                        relPosEncBucketDim
                                        (dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
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
                        withDim @decoderInputEmbedDim $
                          \decoderInputEmbedDim ->
                            withDim @encoderOutputEmbedDim $
                              \encoderOutputEmbedDim ->
                                withDim @ffnDim $
                                  \ffnDim ->
                                    withDim @relPosEncBucketDim @(dropoutP -> Double -> Generator device -> (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)) $
                                      \relPosEncBucketDim -> go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim
    where
      go deviceType dType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP eps = runState $ do
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
        relPosEnc <-
          state $
            withoutDim @headDim
              ( withoutDim @relPosEncBucketDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Embedding ( 'Layout 'Dense) device dataType relPosEncBucketDim headDim 'Nothing)
                          )
                          deviceType
                      )
                      dType
                  )
                  relPosEncBucketDim
              )
              headDim
        pure $ TransformerDecoder decoderStack layerNorm dropout relPosEnc

instance
  ( HasForward
      (Dropout dropoutP)
      decoderInput
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        encoderOutput,
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> decoderRelPosLayout <+> decoderAttentionMaskLayout)
          (device <+> decoderRelPosDevice <+> decoderAttentionMaskDevice)
          (Seq (decoderRelPosDataType <+> 'DataType 'Int64) dataType <+> decoderAttentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ( 'SelectDim ( 'ByIndex 1))
                  ( 'SelectDim ( 'ByIndex 2))
                  ( TransposeF
                      ( 'SelectDim ( 'ByIndex 2))
                      ( 'SelectDim ( 'ByIndex 3))
                      ( EmbeddingF
                          ( 'Shape '[relPosEncBucketDim, headDim])
                          decoderRelPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ( 'SelectDim ( 'ByIndex 1))
                  decoderAttentionMaskShape
              )
          ),
        Tensor
          crossAttentionMaskRequiresGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          ( UnsqueezeF
              ( 'SelectDim ( 'ByIndex 1))
              crossAttentionMaskShape
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      ( LayerNorm
          'WithoutBias
          device
          dataType
          ( 'Shape '[decoderInputEmbedDim])
      )
      stackOutput
      stackGeneratorOutput
      layerNormOutput
      layerNormGeneratorOutput,
    HasForward
      (Dropout dropoutP)
      layerNormOutput
      layerNormGeneratorOutput
      output
      generatorOutput
  ) =>
  HasForward
    (TransformerDecoder numLayers device dataType headDim headEmbedDim embedDim decoderInputEmbedDim encoderOutputEmbedDim ffnDim relPosEncBucketDim dropoutP)
    ( decoderInput,
      encoderOutput,
      Tensor decoderRelPosRequiresGradient decoderRelPosLayout decoderRelPosDevice decoderRelPosDataType decoderRelPosShape,
      Tensor decoderAttentionMaskRequiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor crossAttentionMaskRequiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward TransformerDecoder {..} (decoderInput, encoderOutput, decoderRelPos, decoderAttentionMask, crossAttentionMask) =
    let decoderRelPosBias =
          ireturn decoderRelPos
            >>>= IxState . forward tdRelPosEnc
            >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3))
            >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
        decoderAttentionBias =
          decoderRelPosBias
            >>>= ireturn . (`add` unsqueeze @( 'SelectDim ( 'ByIndex 1)) decoderAttentionMask)
        crossAttentionBias = unsqueeze @( 'SelectDim ( 'ByIndex 1)) crossAttentionMask
     in runIxState $
          ireturn decoderInput
            >>>= IxState . forward tdDropout
            >>>= ( \decoderInput' ->
                     decoderAttentionBias
                       >>>= ( \decoderAttentionBias' ->
                                IxState $
                                  forward
                                    tdStack
                                    ( decoderInput',
                                      encoderOutput,
                                      decoderAttentionBias',
                                      crossAttentionBias
                                    )
                            )
                 )
            >>>= IxState . forward tdLayerNorm
            >>>= IxState . forward tdDropout