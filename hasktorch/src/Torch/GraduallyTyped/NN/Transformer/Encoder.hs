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

module Torch.GraduallyTyped.NN.Transformer.Encoder where

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
import Torch.GraduallyTyped.NN.Transformer.Stack (HasInitializeTransformerStack, HasInitializeTransformerStackC, TransformerStack)
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
  TransformerEncoder
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (inputEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (relPosEncBucketDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerEncoder ::
    forall numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP.
    { -- | encoder layer stack
      teStack :: TransformerStack numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
      -- | encoder layer norm
      teLayerNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[inputEmbedDim]),
      -- | encoder dropout
      teDropout :: Dropout dropoutP,
      -- | relative positional encoding
      teRelPosEnc :: Embedding ( 'Layout 'Dense) device dataType relPosEncBucketDim headDim 'Nothing
    } ->
    TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP

type HasInitializeTransformerEncoderC numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP =
  ( HasInitializeTransformerStackC numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStack
      (1 <=? numLayers)
      numLayers
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      inputEmbedDim
      ffnDim
      dropoutP,
    HasInitializeLayerNormWithoutBiasC device dataType ( 'Shape '[inputEmbedDim]),
    HasInitializeEmbeddingC ( 'Layout 'Dense) device dataType relPosEncBucketDim headDim 'Nothing,
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))))),
    WithDimC inputEmbedDim (WithDimF ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))),
    WithDimC relPosEncBucketDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerEncoderC numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP =>
  HasInitialize (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
  where
  type
    InitializeF (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP) =
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
                                    (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device))
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
                                withDim @relPosEncBucketDim @(dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP, Generator device)) $
                                  \relPosEncBucketDim -> go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim
    where
      go deviceType dType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim
              ( withoutDim @inputEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerStack numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
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
        let layerNorm =
              withoutShape @( 'Shape '[inputEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ( 'Shape '[inputEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [inputEmbedDim]
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
        pure $ TransformerEncoder stack layerNorm dropout relPosEnc

instance
  ( HasForward
      (Dropout dropoutP)
      input
      generator
      dropoutOutput
      dropoutGeneratorOutput,
    HasForward
      (TransformerStack numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim dropoutP)
      ( dropoutOutput,
        Tensor
          'WithGradient
          ( 'Layout 'Dense <+> relPosLayout <+> attentionMaskLayout)
          (device <+> relPosDevice <+> attentionMaskDevice)
          (Seq (relPosDataType <+> 'DataType 'Int64) dataType <+> attentionMaskDataType)
          ( BroadcastShapesF
              ( TransposeF
                  ( 'SelectDim ( 'ByIndex 1))
                  ( 'SelectDim ( 'ByIndex 2))
                  ( TransposeF
                      ( 'SelectDim ( 'ByIndex 2))
                      ( 'SelectDim ( 'ByIndex 3))
                      ( EmbeddingF
                          ( 'Shape '[relPosEncBucketDim, headDim])
                          relPosShape
                      )
                  )
              )
              ( UnsqueezeF
                  ( 'SelectDim ( 'ByIndex 1))
                  attentionMaskShape
              )
          )
      )
      dropoutGeneratorOutput
      stackOutput
      stackGeneratorOutput,
    HasForward
      (LayerNorm 'WithoutBias device dataType ( 'Shape '[inputEmbedDim]))
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
    (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim inputEmbedDim ffnDim relPosEncBucketDim dropoutP)
    ( input,
      Tensor relPosRequiresGradient relPosLayout relPosDevice relPosDataType relPosShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    generator
    output
    generatorOutput
  where
  forward TransformerEncoder {..} (input, relPos, attentionMask) =
    let relPosBias =
          ireturn relPos
            >>>= IxState . forward teRelPosEnc
            >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3))
            >>>= ireturn . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
        attentionBias =
          relPosBias
            >>>= ireturn . (`add` unsqueeze @( 'SelectDim ( 'ByIndex 1)) attentionMask)
     in runIxState $
          ireturn input
            >>>= IxState . forward teDropout
            >>>= (\input' -> attentionBias >>>= (\attentionBias' -> IxState $ forward teStack (input', attentionBias')))
            >>>= IxState . forward teLayerNorm
            >>>= IxState . forward teDropout
