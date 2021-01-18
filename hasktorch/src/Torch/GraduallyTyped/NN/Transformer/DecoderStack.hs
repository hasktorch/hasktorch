{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -v2
                -fomit-interface-pragmas
                -fplugin TypeLevel.Rewrite
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyRightAssociativeL
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL1
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL2C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL3C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL4C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL5C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL6C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL7C
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8
                -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Unify.UnifyIdempotenceL8C #-}

module Torch.GraduallyTyped.NN.Transformer.DecoderStack where

import Control.Monad.Indexed (ireturn, (>>>=))
import Control.Monad.Indexed.State (IxState (..))
import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense))
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Transformer.CrossAttention (CrossAttentionOutputShape)
import Torch.GraduallyTyped.NN.Transformer.DecoderBlock (HasInitializeTransformerDecoderBlockC, TransformerDecoderBlock)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (FeedForwardNetworkOutputShape)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttentionOutputShape)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  TransformerDecoderStack
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerDecoderStackNil ::
    TransformerDecoderStack 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  TransformerDecoderStackCons ::
    TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
    TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
    TransformerDecoderStack (numLayers + 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP

class
  HasInitializeTransformerDecoderStack
    (isCons :: Bool)
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  initializeTransformerDecoderStack ::
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
                              keyEmbedDim
                              ( WithDimF
                                  ffnDim
                                  (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
                              )
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerDecoderStackC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerDecoderStackC 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
  HasInitializeTransformerDecoderStack 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  where
  initializeTransformerDecoderStack =
    withDevice @device $
      \_deviceType ->
        withDataType @dataType $
          \_dType ->
            withDim @headDim $
              \_headDim ->
                withDim @headEmbedDim $
                  \_headEmbedDim ->
                    withDim @embedDim $
                      \_embedDim ->
                        withDim @queryEmbedDim $
                          \_queryEmbedDim ->
                            withDim @keyEmbedDim $
                              \_keyEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
                                  \_ffnDim _dropoutP _eps g -> (TransformerDecoderStackNil, g)

instance
  ( HasInitializeTransformerDecoderBlockC device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderStackC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitializeTransformerDecoderStackC (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerDecoderStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  ) =>
  HasInitializeTransformerDecoderStack 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP
  where
  initializeTransformerDecoderStack =
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
                            withDim @keyEmbedDim $
                              \keyEmbedDim ->
                                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device)) $
                                  \ffnDim -> go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP eps = runState $ do
        decoderStack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerDecoderStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoderStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
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
                  keyEmbedDim
              )
              ffnDim
              dropoutP
              eps
        decoderBlock <-
          state $
            withoutDim @ffnDim
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDim @headEmbedDim
                              ( withoutDim @headDim
                                  ( withoutDataType @dataType
                                      ( withoutDevice @device
                                          ( initialize @(TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
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
                  keyEmbedDim
              )
              ffnDim
              dropoutP
              eps
        pure $ TransformerDecoderStackCons decoderBlock decoderStack

instance
  HasInitializeTransformerDecoderStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) =
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
                                keyEmbedDim
                                ( WithDimF
                                    ffnDim
                                    (dropoutP -> Double -> Generator device -> (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP, Generator device))
                                )
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerDecoderStack @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @keyEmbedDim @ffnDim @dropoutP

class
  HasForwardTransformerDecoderStack
    (isCons :: Bool)
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
    (requiresGradient :: RequiresGradient)
    (queryLayout :: Layout LayoutType)
    (queryDevice :: Device (DeviceType Nat))
    (queryDataType :: DataType DType)
    (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (keyLayout :: Layout LayoutType)
    (keyDevice :: Device (DeviceType Nat))
    (keyDataType :: DataType DType)
    (keyShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (decoderAttentionMaskLayout :: Layout LayoutType)
    (decoderAttentionMaskDevice :: Device (DeviceType Nat))
    (decoderAttentionMaskDataType :: DataType DType)
    (decoderAttentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (crossAttentionMaskLayout :: Layout LayoutType)
    (crossAttentionMaskDevice :: Device (DeviceType Nat))
    (crossAttentionMaskDataType :: DataType DType)
    (crossAttentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (generatorDevice :: Device (DeviceType Nat))
  where
  type HasForwardTransformerDecoderStackOutput isCons numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice :: Type
  type HasForwardTransformerDecoderStackGeneratorOutput isCons numLayers device queryDevice keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice generatorDevice :: Type
  forwardTransformerDecoderStack ::
    TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP ->
    ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
      Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
      Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
    ) ->
    Generator generatorDevice ->
    ( HasForwardTransformerDecoderStackOutput isCons numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice,
      HasForwardTransformerDecoderStackGeneratorOutput isCons numLayers device queryDevice keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice generatorDevice
    )

instance HasForwardTransformerDecoderStack 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice where
  type HasForwardTransformerDecoderStackOutput 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice = Tensor requiresGradient queryLayout queryDevice queryDataType queryShape
  type HasForwardTransformerDecoderStackGeneratorOutput 'False 0 device queryDevice keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice generatorDevice = Generator generatorDevice
  forwardTransformerDecoderStack TransformerDecoderStackNil (query, _key, _decoderAttentionMask, _crossAttentionMask) g = (query, g)

instance
  ( HasForward
      (TransformerDecoderBlock device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor requiresGradient keyLayout keyDevice keyDataType keyShape,
        Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape,
        Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape
      )
      (Generator generatorDevice),
    HasForward
      (TransformerDecoderStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
      ( Tensor
          requiresGradient
          (queryLayout <+> 'Layout 'Dense <+> decoderAttentionMaskLayout <+> keyLayout <+> crossAttentionMaskLayout)
          (queryDevice <+> device <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice)
          (queryDataType <+> dataType <+> decoderAttentionMaskDataType <+> keyDataType <+> crossAttentionMaskDataType)
          ( FeedForwardNetworkOutputShape
              queryEmbedDim
              ffnDim
              ( CrossAttentionOutputShape
                  headDim
                  headEmbedDim
                  embedDim
                  queryEmbedDim
                  keyEmbedDim
                  ( SelfAttentionOutputShape
                      headDim
                      headEmbedDim
                      embedDim
                      queryEmbedDim
                      queryShape
                      decoderAttentionMaskShape
                  )
                  keyShape
                  crossAttentionMaskShape
              )
          ),
        Tensor
          requiresGradient
          keyLayout
          keyDevice
          keyDataType
          keyShape,
        Tensor
          requiresGradient
          decoderAttentionMaskLayout
          decoderAttentionMaskDevice
          decoderAttentionMaskDataType
          decoderAttentionMaskShape,
        Tensor
          requiresGradient
          crossAttentionMaskLayout
          crossAttentionMaskDevice
          crossAttentionMaskDataType
          crossAttentionMaskShape
      )
      (Generator (device <+> queryDevice <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice))
  ) =>
  HasForwardTransformerDecoderStack 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice
  where
  type
    HasForwardTransformerDecoderStackOutput 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice =
      HasForwardTransformerDecoderStackOutput
        (1 <=? numLayers - 1)
        (numLayers - 1)
        device
        dataType
        headDim
        headEmbedDim
        embedDim
        queryEmbedDim
        keyEmbedDim
        ffnDim
        requiresGradient
        (queryLayout <+> 'Layout 'Dense <+> decoderAttentionMaskLayout <+> keyLayout <+> crossAttentionMaskLayout)
        (queryDevice <+> device <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice)
        (queryDataType <+> dataType <+> decoderAttentionMaskDataType <+> keyDataType <+> crossAttentionMaskDataType)
        ( FeedForwardNetworkOutputShape
            queryEmbedDim
            ffnDim
            ( CrossAttentionOutputShape
                headDim
                headEmbedDim
                embedDim
                queryEmbedDim
                keyEmbedDim
                ( SelfAttentionOutputShape
                    headDim
                    headEmbedDim
                    embedDim
                    queryEmbedDim
                    queryShape
                    decoderAttentionMaskShape
                )
                keyShape
                crossAttentionMaskShape
            )
        )
        keyLayout
        keyDevice
        keyDataType
        keyShape
        decoderAttentionMaskLayout
        decoderAttentionMaskDevice
        decoderAttentionMaskDataType
        decoderAttentionMaskShape
        crossAttentionMaskLayout
        crossAttentionMaskDevice
        crossAttentionMaskDataType
        crossAttentionMaskShape
        (device <+> queryDevice <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice)
  type HasForwardTransformerDecoderStackGeneratorOutput 'True numLayers device queryDevice keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice generatorDevice = HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers - 1) (numLayers - 1) device (queryDevice <+> device <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice) keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice (device <+> queryDevice <+> generatorDevice <+> decoderAttentionMaskDevice <+> keyDevice <+> crossAttentionMaskDevice)
  forwardTransformerDecoderStack (TransformerDecoderStackCons decoderBlock decoderStack) (query, key, decoderAttentionMask, crossAttentionMask) =
    runIxState $
      ireturn (query, key, decoderAttentionMask, crossAttentionMask)
        >>>= IxState . forward decoderBlock
        >>>= ( \query' ->
                 IxState $
                   forward
                     @(TransformerDecoderStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP)
                     decoderStack
                     (query', key, decoderAttentionMask, crossAttentionMask)
             )

instance
  HasForwardTransformerDecoderStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice =>
  HasForward (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient keyLayout keyDevice keyDataType keyShape, Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice)
  where
  type ForwardOutput (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient keyLayout keyDevice keyDataType keyShape, Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice) = HasForwardTransformerDecoderStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape generatorDevice
  type ForwardGeneratorOutput (TransformerDecoderStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim keyEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient keyLayout keyDevice keyDataType keyShape, Tensor requiresGradient decoderAttentionMaskLayout decoderAttentionMaskDevice decoderAttentionMaskDataType decoderAttentionMaskShape, Tensor requiresGradient crossAttentionMaskLayout crossAttentionMaskDevice crossAttentionMaskDataType crossAttentionMaskShape) (Generator generatorDevice) = HasForwardTransformerDecoderStackGeneratorOutput (1 <=? numLayers) numLayers device queryDevice keyDevice decoderAttentionMaskDevice crossAttentionMaskDevice generatorDevice
  forward = forwardTransformerDecoderStack @(1 <=? numLayers)
