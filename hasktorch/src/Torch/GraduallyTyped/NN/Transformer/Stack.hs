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
{-# OPTIONS_GHC -fomit-interface-pragmas
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

module Torch.GraduallyTyped.NN.Transformer.Stack where

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
import Torch.GraduallyTyped.NN.Transformer.Block (HasInitializeTransformerBlockC, TransformerBlock)
import Torch.GraduallyTyped.NN.Transformer.FeedForwardNetwork (FeedForwardNetworkOutputShape)
import Torch.GraduallyTyped.NN.Transformer.SelfAttention (SelfAttentionOutputShape)
import Torch.GraduallyTyped.Random (Generator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient)
import Torch.GraduallyTyped.Shape (Dim (..), Name (..), Shape (..), Size (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Unify (type (<+>))

data
  TransformerStack
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
  TransformerStackNil ::
    forall device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    TransformerStack 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  TransformerStackCons ::
    forall numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP.
    TransformerBlock device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
    TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
    TransformerStack (numLayers + 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP

class
  HasInitializeTransformerStack
    (isCons :: Bool)
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
  initializeTransformerStack ::
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
                              (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerStackC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)))),
    WithDimC queryEmbedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerStackC 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
  HasInitializeTransformerStack 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  where
  initializeTransformerStack =
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
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \_ffnDim _dropoutP _eps g -> (TransformerStackNil, g)

instance
  ( HasInitializeTransformerBlockC device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitializeTransformerStackC (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP,
    HasInitialize (TransformerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  ) =>
  HasInitializeTransformerStack 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP
  where
  initializeTransformerStack =
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
                            withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device)) $
                              \ffnDim -> go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim
    where
      go deviceType dType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP eps = runState $ do
        stack <-
          state $
            withoutDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
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
        block <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerBlock device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
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
        pure $ TransformerStackCons block stack

instance
  HasInitializeTransformerStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP =>
  HasInitialize (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) =
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
                                (dropoutP -> Double -> Generator device -> (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP, Generator device))
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerStack @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @queryEmbedDim @ffnDim @dropoutP

class
  HasForwardTransformerStack
    (isCons :: Bool)
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
    (requiresGradient :: RequiresGradient)
    (queryLayout :: Layout LayoutType)
    (queryDevice :: Device (DeviceType Nat))
    (queryDataType :: DataType DType)
    (queryShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (attentionMaskLayout :: Layout LayoutType)
    (attentionMaskDevice :: Device (DeviceType Nat))
    (attentionMaskDataType :: DataType DType)
    (attentionMaskShape :: Shape [Dim (Name Symbol) (Size Nat)])
    (generatorDevice :: Device (DeviceType Nat))
  where
  type HasForwardTransformerStackOutput isCons numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice :: Type
  type HasForwardTransformerStackGeneratorOutput isCons numLayers device queryDevice attentionMaskDevice generatorDevice :: Type
  forwardTransformerStack ::
    TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP ->
    ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
      Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
    ) ->
    Generator generatorDevice ->
    ( HasForwardTransformerStackOutput isCons numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice,
      HasForwardTransformerStackGeneratorOutput isCons numLayers device queryDevice attentionMaskDevice generatorDevice
    )

instance HasForwardTransformerStack 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice where
  type HasForwardTransformerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice = Tensor requiresGradient queryLayout queryDevice queryDataType queryShape
  type HasForwardTransformerStackGeneratorOutput 'False 0 device queryDevice attentionMaskDevice generatorDevice = Generator generatorDevice
  forwardTransformerStack TransformerStackNil (query, _attentionMask) g = (query, g)

instance
  ( HasForward
      (TransformerBlock device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      ( Tensor requiresGradient queryLayout queryDevice queryDataType queryShape,
        Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape
      )
      (Generator generatorDevice),
    HasForward
      (TransformerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
      ( Tensor
          requiresGradient
          (queryLayout <+> 'Layout 'Dense <+> attentionMaskLayout)
          (queryDevice <+> device <+> generatorDevice <+> attentionMaskDevice)
          (queryDataType <+> dataType <+> attentionMaskDataType)
          ( FeedForwardNetworkOutputShape
              queryEmbedDim
              ffnDim
              (SelfAttentionOutputShape headDim headEmbedDim embedDim queryEmbedDim queryShape attentionMaskShape)
          ),
        Tensor
          requiresGradient
          attentionMaskLayout
          attentionMaskDevice
          attentionMaskDataType
          attentionMaskShape
      )
      (Generator (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice))
  ) =>
  HasForwardTransformerStack 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  where
  type
    HasForwardTransformerStackOutput 'True numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice =
      HasForwardTransformerStackOutput
        (1 <=? numLayers - 1)
        (numLayers - 1)
        device
        dataType
        headDim
        headEmbedDim
        embedDim
        queryEmbedDim
        ffnDim
        requiresGradient
        (queryLayout <+> 'Layout 'Dense <+> attentionMaskLayout)
        (queryDevice <+> device <+> generatorDevice <+> attentionMaskDevice)
        (queryDataType <+> dataType <+> attentionMaskDataType)
        ( FeedForwardNetworkOutputShape
            queryEmbedDim
            ffnDim
            (SelfAttentionOutputShape headDim headEmbedDim embedDim queryEmbedDim queryShape attentionMaskShape)
        )
        attentionMaskLayout
        attentionMaskDevice
        attentionMaskDataType
        attentionMaskShape
        (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice)
  type HasForwardTransformerStackGeneratorOutput 'True numLayers device queryDevice attentionMaskDevice generatorDevice = HasForwardTransformerStackGeneratorOutput (1 <=? numLayers - 1) (numLayers - 1) device (queryDevice <+> device <+> generatorDevice <+> attentionMaskDevice) attentionMaskDevice (device <+> queryDevice <+> generatorDevice <+> attentionMaskDevice)
  forwardTransformerStack (TransformerStackCons block stack) (query, attentionMask) =
    runIxState $
      ireturn (query, attentionMask)
        >>>= IxState . forward block
        >>>= ( \query' ->
                 IxState $
                   forward
                     @(TransformerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP)
                     stack
                     (query', attentionMask)
             )

instance
  HasForwardTransformerStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice =>
  HasForward (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice)
  where
  type ForwardOutput (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice) = HasForwardTransformerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim requiresGradient queryLayout queryDevice queryDataType queryShape attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape generatorDevice
  type ForwardGeneratorOutput (TransformerStack numLayers device dataType headDim headEmbedDim embedDim queryEmbedDim ffnDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape, Tensor requiresGradient attentionMaskLayout attentionMaskDevice attentionMaskDataType attentionMaskShape) (Generator generatorDevice) = HasForwardTransformerStackGeneratorOutput (1 <=? numLayers) numLayers device queryDevice attentionMaskDevice generatorDevice
  forward = forwardTransformerStack @(1 <=? numLayers)