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

module Torch.GraduallyTyped.NN.Transformer.Encoder where

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
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormWithoutBiasC, LayerNorm, LayerNormHasBias (..))
import Torch.GraduallyTyped.NN.Transformer.Stack (HasForwardTransformerStack, HasForwardTransformerStackGeneratorOutput, HasForwardTransformerStackOutput, HasInitializeTransformerStack, HasInitializeTransformerStackC, TransformerStack)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..), WithShapeC (..))
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
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerEncoder ::
    forall numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    { -- | encoder layer stack
      teStack :: TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
      -- | encoder layer norm
      teLayerNorm :: LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]),
      -- | encoder dropout
      teDropout :: Dropout dropoutP
    } ->
    TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

type HasInitializeTransformerEncoderC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =
  ( HasInitializeTransformerStackC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStack
      (1 <=? numLayers)
      numLayers
      device
      dataType
      headDim
      headEmbedDim
      embedDim
      queryEmbedDim
      ffnDim
      dropoutP,
    HasInitializeLayerNormWithoutBiasC device dataType ( 'Shape '[queryEmbedDim]),
    Scalar dropoutP,
    WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerEncoderC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
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
                            queryEmbedDim
                            ( WithDimF
                                ffnDim
                                (dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
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
                        withDim @queryEmbedDim $
                          \queryEmbedDim ->
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \ffnDim -> go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
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
                  queryEmbedDim
              )
              ffnDim
              dropoutP
              eps
        let layerNorm =
              withoutShape @( 'Shape '[queryEmbedDim])
                ( withoutDataType @dataType
                    ( withoutDevice @device
                        ( initialize @(LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
                        )
                        deviceType
                    )
                    dType
                )
                [queryEmbedDim]
                eps
        let dropout = initialize @(Dropout dropoutP) dropoutP
        pure $ TransformerEncoder stack layerNorm dropout

type HasForwardTransformerEncoderC
  (numLayers :: Nat)
  (device :: Device (DeviceType Nat))
  (dataType :: DataType DType)
  (headDim :: Dim (Name Symbol) (Size Nat))
  (headEmbedDim :: Dim (Name Symbol) (Size Nat))
  (embedDim :: Dim (Name Symbol) (Size Nat))
  (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
  (ffnDim :: Dim (Name Symbol) (Size Nat))
  (dropoutP :: Type)
  (inputRequiresGradient :: RequiresGradient)
  (inputLayout :: Layout LayoutType)
  (inputDevice :: Device (DeviceType Nat))
  (inputDataType :: DataType DType)
  (inputShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (attentionMaskRequiresGradient :: RequiresGradient)
  (attentionMaskLayout :: Layout LayoutType)
  (attentionMaskDevice :: Device (DeviceType Nat))
  (attentionMaskDataType :: DataType DType)
  (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
  (generatorDevice :: Device (DeviceType Nat)) =
  ( Scalar dropoutP,
    HasForwardTransformerStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice),
    HasForward
      (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
      (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
      (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice)),
    HasForward
      (Dropout dropoutP)
      ( ForwardOutput
          (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
          (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
          (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
      )
      ( ForwardGeneratorOutput
          (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
          (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
          (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
      )
  )

instance
  HasForwardTransformerEncoderC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP inputRequiresGradient inputLayout inputDevice inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice =>
  HasForward
    (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
    ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
      Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    )
    (Generator generatorDevice)
  where
  type
    ForwardOutput
      (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardOutput
        (Dropout dropoutP)
        ( ForwardOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
            (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
            (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
        )
        ( ForwardGeneratorOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
            (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
            (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
        )
  type
    ForwardGeneratorOutput
      (TransformerEncoder numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      ( Tensor inputRequiresGradient inputLayout inputDevice inputDataType inputShape,
        Tensor attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice) =
      ForwardGeneratorOutput
        (Dropout dropoutP)
        ( ForwardOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
            (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
            (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
        )
        ( ForwardGeneratorOutput
            (LayerNorm 'WithoutBias device dataType ( 'Shape '[queryEmbedDim]))
            (HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim inputRequiresGradient inputLayout (inputDevice <+> generatorDevice) inputDataType inputShape attentionMaskRequiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape (inputDevice <+> generatorDevice))
            (HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device (inputDevice <+> generatorDevice) attentionMaskDevice (inputDevice <+> generatorDevice))
        )
  forward TransformerEncoder {..} (input, attentionMask) =
    runIxState $
      ireturn input
        >>>= IxState . forward teDropout
        >>>= (\input' -> IxState $ forward teStack (input', attentionMask))
        >>>= IxState . forward teLayerNorm
        >>>= IxState . forward teDropout