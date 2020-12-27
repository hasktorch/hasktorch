{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
-- {-# OPTIONS_GHC -fplugin TypeLevel.Rewrite
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL1
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL2
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL3
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL4
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL5
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL6
--                 -fplugin-opt=TypeLevel.Rewrite:Torch.GraduallyTyped.Device.UnifyDeviceL7 #-}

module Torch.GraduallyTyped.NN.Transformer where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat, Symbol, type (+), type (-), type (<=?))
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (DataType (DataType), UnifyDataTypeF, WithDataTypeC (..))
import Torch.GraduallyTyped.Device (Device (..), DeviceType (..), UnifyDeviceF, UnifyDeviceL, WithDeviceC (..))
import Torch.GraduallyTyped.Layout (Layout (Layout), LayoutType (Dense), UnifyLayoutF)
import Torch.GraduallyTyped.NN.Class (HasForward (..), HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout)
import Torch.GraduallyTyped.NN.Functional.Activation (relu)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.NN.Functional.Normalization (LayerNormF)
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearC, Linear(..))
import Torch.GraduallyTyped.NN.Normalization (HasInitializeLayerNormC, LayerNorm)
import Torch.GraduallyTyped.Random (Generator, mkGenerator)
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient (Dependent))
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (KnownShape, BroadcastShapesF, By (..), Dim (..), KnownDim (..), Name (..), SelectDim (..), Shape (..), Size (..), UnifyDimF, WithDimC (..), WithShapeC (..), dimSize, type (!))
import Torch.GraduallyTyped.Tensor.Creation (randn)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, reshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (add, divScalar)
import Torch.GraduallyTyped.Tensor.Type (shape, Tensor)

data
  MultiheadAttention
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  MultiheadAttention ::
    { -- | in-projection for query
      mhaQInProj :: Linear device dataType queryEmbedDim embedDim,
      -- | in-projection for key
      mhaKInProj :: Linear device dataType keyEmbedDim embedDim,
      -- | in-projection for value
      mhaVInProj :: Linear device dataType valueEmbedDim embedDim,
      -- | out-projection
      mhaOutProj :: Linear device dataType embedDim queryEmbedDim,
      -- | dropout
      mhaDropout :: Dropout dropoutP
    } ->
    MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDimC embedDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    WithDimC queryEmbedDim (Generator device -> (Linear device dataType embedDim queryEmbedDim, Generator device)),
    HasInitializeLinearC device dataType queryEmbedDim embedDim,
    HasInitializeLinearC device dataType keyEmbedDim embedDim,
    HasInitializeLinearC device dataType valueEmbedDim embedDim,
    HasInitializeLinearC device dataType embedDim queryEmbedDim,
    Scalar dropoutP
  )

instance
  HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    queryEmbedDim
                    ( WithDimF
                        keyEmbedDim
                        ( WithDimF
                            valueEmbedDim
                            (dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
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
            withDim @embedDim $
              \embedDim ->
                withDim @queryEmbedDim $
                  \queryEmbedDim ->
                    withDim @keyEmbedDim $
                      \keyEmbedDim ->
                        withDim @valueEmbedDim @(dropoutP -> Generator device -> (MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                          \valueEmbedDim ->
                            go deviceType dType embedDim queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP = runState $ do
        qInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType queryEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              embedDim
        kInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType keyEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  keyEmbedDim
              )
              embedDim
        vInProj <-
          state $
            withoutDim @embedDim
              ( withoutDim @valueEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType valueEmbedDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  valueEmbedDim
              )
              embedDim
        outProj <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @embedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType embedDim queryEmbedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  embedDim
              )
              queryEmbedDim
        dropout <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        pure $ MultiheadAttention qInProj kInProj vInProj outProj dropout

type BatchDim queryShape keyShape valueShape = UnifyDimF (UnifyDimF (queryShape ! 0) (keyShape ! 0)) (valueShape ! 0)

getBatchDim :: [Dim String Integer] -> [Dim String Integer] -> [Dim String Integer] -> Dim String Integer 
getBatchDim (batchDim : _queryDims) (batchDim' : _keyDims) (batchDim'' : _valueDims) | batchDim == batchDim' && batchDim' == batchDim'' = batchDim
getBatchDim _ _ _ = error "batchDim"

type QuerySeqDim queryShape = queryShape ! 1

getQuerySeqDim :: [Dim String Integer] -> Dim String Integer
getQuerySeqDim (_batchDim : querySeqDim : _queryDims) = querySeqDim
getQuerySeqDim _ = error "querySeqDim"

type KeySeqDim keyShape valueShape = UnifyDimF (keyShape ! 1) (valueShape ! 1)

getKeySeqDim :: [Dim String Integer] -> [Dim String Integer] -> Dim String Integer
getKeySeqDim (_batchDim : keySeqDim : _keyDims) (_batchDim' : keySeqDim' : _valueDims) | keySeqDim == keySeqDim' = keySeqDim
getKeySeqDim _ _ = error "keySeqDim"

getEmbedDim ::
  forall device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
  (KnownDim embedDim, KnownDim queryEmbedDim) =>
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  Dim String Integer
getEmbedDim MultiheadAttention {..} = case dimVal @embedDim of
  Dim (Name name) (Size size) -> Dim name size
  Dim _ _ -> head . shape . linearWeight $ mhaQInProj

type MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputGeneratorDevice outputDataType outputShape =
  ( KnownDim embedDim,
    KnownDim headEmbedDim,
    KnownDim headDim,
    KnownDim keySeqDim,
    KnownDim querySeqDim,
    KnownDim batchDim,
    KnownShape queryShape,
    KnownShape keyShape,
    KnownShape valueShape,
    Scalar dropoutP,
    WithDimC
      headDim
      ( WithDimF
          headEmbedDim
          ( Generator generatorDevice ->
            ( Tensor
                requiresGradient
                outputLayout
                outputDevice
                outputDataType
                outputShape,
              Generator outputGeneratorDevice
            )
          )
      ),
    WithDimC
      headEmbedDim
      ( Generator generatorDevice ->
        ( Tensor
            requiresGradient
            outputLayout
            outputDevice
            outputDataType
            outputShape,
          Generator outputGeneratorDevice
        )
      ),
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF keyLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, keyEmbedDim])
              ( 'Shape '[embedDim])
              keyShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF keyLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, keyEmbedDim])
                  ( 'Shape '[embedDim])
                  keyShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF valueLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, valueEmbedDim])
              ( 'Shape '[embedDim])
              valueShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF valueLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, valueEmbedDim])
                  ( 'Shape '[embedDim])
                  valueShape
              )
              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
      ( Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF queryLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
          ( LinearF
              ( 'Shape '[embedDim, queryEmbedDim])
              ( 'Shape '[embedDim])
              queryShape
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              (UnifyLayoutF queryLayout ( 'Layout 'Dense))
              ( 'Layout 'Dense)
          )
          (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
          (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
          ( ReshapeF
              ( LinearF
                  ( 'Shape '[embedDim, queryEmbedDim])
                  ( 'Shape '[embedDim])
                  queryShape
              )
              ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
          )
      ),
    WithShapeC
      ( 'Shape '[batchDim, querySeqDim, embedDim])
      ( Tensor
          requiresGradient
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( UnifyDeviceF
              ( UnifyDeviceF
                  ( UnifyDeviceF
                      (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
                      (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
                  )
                  generatorDevice
              )
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          ( UnifyDataTypeF
              ( UnifyDataTypeF
                  (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
                  (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
              )
              (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
          )
          ( TransposeF
              ( 'SelectDim ( 'ByIndex 1))
              ( 'SelectDim ( 'ByIndex 2))
              ( MatmulF
                  ( SoftmaxF
                      ( 'SelectDim ( 'ByIndex 3))
                      ( MatmulF
                          ( TransposeF
                              ( 'SelectDim ( 'ByIndex 1))
                              ( 'SelectDim ( 'ByIndex 2))
                              ( ReshapeF
                                  ( LinearF
                                      ( 'Shape '[embedDim, queryEmbedDim])
                                      ( 'Shape '[embedDim])
                                      queryShape
                                  )
                                  ( 'Shape
                                      '[batchDim, querySeqDim, headDim, headEmbedDim]
                                  )
                              )
                          )
                          ( TransposeF
                              ( 'SelectDim ( 'ByIndex 2))
                              ( 'SelectDim ( 'ByIndex 3))
                              ( TransposeF
                                  ( 'SelectDim ( 'ByIndex 1))
                                  ( 'SelectDim ( 'ByIndex 2))
                                  ( ReshapeF
                                      ( LinearF
                                          ( 'Shape '[embedDim, keyEmbedDim])
                                          ( 'Shape '[embedDim])
                                          keyShape
                                      )
                                      ( 'Shape
                                          '[ batchDim,
                                             keySeqDim,
                                             headDim,
                                             headEmbedDim
                                           ]
                                      )
                                  )
                              )
                          )
                      )
                  )
                  ( TransposeF
                      ( 'SelectDim ( 'ByIndex 1))
                      ( 'SelectDim ( 'ByIndex 2))
                      ( ReshapeF
                          ( LinearF
                              ( 'Shape '[embedDim, valueEmbedDim])
                              ( 'Shape '[embedDim])
                              valueShape
                          )
                          ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                      )
                  )
              )
          ) ->
        Tensor
          requiresGradient
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( UnifyDeviceF
              ( UnifyDeviceF
                  ( UnifyDeviceF
                      (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
                      (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
                  )
                  generatorDevice
              )
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          ( UnifyDataTypeF
              ( UnifyDataTypeF
                  ( UnifyDataTypeF
                      (UnifyDataTypeF queryDataType dataType)
                      dataType
                  )
                  ( UnifyDataTypeF
                      (UnifyDataTypeF keyDataType dataType)
                      dataType
                  )
              )
              ( UnifyDataTypeF
                  (UnifyDataTypeF valueDataType dataType)
                  dataType
              )
          )
          ( ReshapeF
              ( TransposeF
                  ( 'SelectDim ( 'ByIndex 1))
                  ( 'SelectDim ( 'ByIndex 2))
                  ( MatmulF
                      ( SoftmaxF
                          ( 'SelectDim ( 'ByIndex 3))
                          ( MatmulF
                              ( TransposeF
                                  ( 'SelectDim ( 'ByIndex 1))
                                  ( 'SelectDim ( 'ByIndex 2))
                                  ( ReshapeF
                                      ( LinearF
                                          ( 'Shape '[embedDim, queryEmbedDim])
                                          ( 'Shape '[embedDim])
                                          queryShape
                                      )
                                      ( 'Shape
                                          '[ batchDim,
                                             querySeqDim,
                                             headDim,
                                             headEmbedDim
                                           ]
                                      )
                                  )
                              )
                              ( TransposeF
                                  ( 'SelectDim ( 'ByIndex 2))
                                  ( 'SelectDim ( 'ByIndex 3))
                                  ( TransposeF
                                      ( 'SelectDim ( 'ByIndex 1))
                                      ( 'SelectDim ( 'ByIndex 2))
                                      ( ReshapeF
                                          ( LinearF
                                              ( 'Shape '[embedDim, keyEmbedDim])
                                              ( 'Shape '[embedDim])
                                              keyShape
                                          )
                                          ( 'Shape
                                              '[ batchDim,
                                                 keySeqDim,
                                                 headDim,
                                                 headEmbedDim
                                               ]
                                          )
                                      )
                                  )
                              )
                          )
                      )
                      ( TransposeF
                          ( 'SelectDim ( 'ByIndex 1))
                          ( 'SelectDim ( 'ByIndex 2))
                          ( ReshapeF
                              ( LinearF
                                  ( 'Shape '[embedDim, valueEmbedDim])
                                  ( 'Shape '[embedDim])
                                  valueShape
                              )
                              ( 'Shape
                                  '[batchDim, keySeqDim, headDim, headEmbedDim]
                              )
                          )
                      )
                  )
              )
              ( 'Shape '[batchDim, querySeqDim, embedDim])
          )
      ),
    -- UnifyDeviceL queryDevice device,
    -- UnifyDeviceL keyDevice device,
    -- UnifyDeviceL valueDevice device,
    batchDim ~ BatchDim queryShape keyShape valueShape,
    querySeqDim ~ QuerySeqDim queryShape,
    keySeqDim ~ KeySeqDim keyShape valueShape
  )

type MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice =
  -- UnifyDeviceF valueDevice (UnifyDeviceF generatorDevice (UnifyDeviceF keyDevice (UnifyDeviceF queryDevice device)))
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
                    (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
                )
                generatorDevice
            )
            (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
        )
        device
    )
    device

type MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice =
  -- UnifyDeviceF generatorDevice (UnifyDeviceF keyDevice (UnifyDeviceF queryDevice device))
  UnifyDeviceF
    ( UnifyDeviceF
        (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
        (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
    )
    generatorDevice

type MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape =
  LinearF
    ( 'Shape '[queryEmbedDim, embedDim])
    ( 'Shape '[queryEmbedDim])
    ( ReshapeF
        ( TransposeF
            ( 'SelectDim ( 'ByIndex 1))
            ( 'SelectDim ( 'ByIndex 2))
            ( MatmulF
                ( SoftmaxF
                    ( 'SelectDim ( 'ByIndex 3))
                    ( MatmulF
                        ( TransposeF
                            ( 'SelectDim ( 'ByIndex 1))
                            ( 'SelectDim ( 'ByIndex 2))
                            ( ReshapeF
                                ( LinearF
                                    ( 'Shape '[embedDim, queryEmbedDim])
                                    ( 'Shape '[embedDim])
                                    queryShape
                                )
                                ( 'Shape '[batchDim, querySeqDim, headDim, headEmbedDim])
                            )
                        )
                        ( TransposeF
                            ( 'SelectDim ( 'ByIndex 2))
                            ( 'SelectDim ( 'ByIndex 3))
                            ( TransposeF
                                ( 'SelectDim ( 'ByIndex 1))
                                ( 'SelectDim ( 'ByIndex 2))
                                ( ReshapeF
                                    ( LinearF
                                        ( 'Shape '[embedDim, keyEmbedDim])
                                        ( 'Shape '[embedDim])
                                        keyShape
                                    )
                                    ( 'Shape
                                        '[batchDim, keySeqDim, headDim, headEmbedDim]
                                    )
                                )
                            )
                        )
                    )
                )
                ( TransposeF
                    ( 'SelectDim ( 'ByIndex 1))
                    ( 'SelectDim ( 'ByIndex 2))
                    ( ReshapeF
                        ( LinearF
                            ( 'Shape '[embedDim, valueEmbedDim])
                            ( 'Shape '[embedDim])
                            valueShape
                        )
                        ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                    )
                )
            )
        )
        ( 'Shape '[batchDim, querySeqDim, embedDim])
    )

type MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            ( UnifyDataTypeF
                (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
                (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
            )
            (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
        )
        dataType
    )
    dataType

type MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout = 
  UnifyLayoutF
      ( UnifyLayoutF
          ( UnifyLayoutF
              ( UnifyLayoutF
                  ( UnifyLayoutF
                      (UnifyLayoutF queryLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
                  ( UnifyLayoutF
                      (UnifyLayoutF keyLayout ( 'Layout 'Dense))
                      ( 'Layout 'Dense)
                  )
              )
              ( UnifyLayoutF
                  (UnifyLayoutF valueLayout ( 'Layout 'Dense))
                  ( 'Layout 'Dense)
              )
          )
          ( 'Layout 'Dense)
      )
      ( 'Layout 'Dense)

multiheadAttention ::
  forall headDim headEmbedDim generatorDevice batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape outputLayout outputDevice outputGeneratorDevice outputDataType outputShape.
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputGeneratorDevice outputDataType outputShape,
    outputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout,
    outputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice,
    outputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    outputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType,
    outputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape
  ) =>
  -- | multi-head attention model
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  WithDimF
    headDim
    ( WithDimF
        headEmbedDim
        ( Generator generatorDevice ->
          ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
            Generator outputGeneratorDevice
          )
        )
    )
multiheadAttention MultiheadAttention {..} query key value =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let batchDim = case dimVal @batchDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> getBatchDim (shape query) (shape key) (shape value)
          querySeqDim = case dimVal @querySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> getQuerySeqDim (shape query)
          keySeqDim = case dimVal @keySeqDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> getKeySeqDim (shape key) (shape value)
          embedDim = case dimVal @embedDim of
            Dim (Name name) (Size size) -> Dim name size
            Dim _ _ -> undefined
          scaling :: Double = sqrt . fromIntegral . dimSize $ headDim
          (q, g') =
            let (query', g') = forward mhaQInProj query g
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @querySeqDim @headDim @headEmbedDim [batchDim, querySeqDim, headDim, headEmbedDim]
                    . flip divScalar scaling
                    $ query',
                  g'
                )
          (k, g'') =
            let (key', g'') = forward mhaKInProj key g'
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
                    $ key',
                  g''
                )
          qk = q `matmul` transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3)) k
          (weights, g''') = forward @_ @_ @(Generator generatorDevice) mhaDropout (softmax @( 'SelectDim ( 'ByIndex 3)) qk) g''
          (v, g'''') =
            let (value', g'''') = forward mhaVInProj value g'''
             in ( transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
                    . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
                    $ value',
                  g''''
                )
          weights' =
            reshape'' @batchDim @querySeqDim @embedDim [batchDim, querySeqDim, embedDim]
              . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
              $ weights `matmul` v
       in forward mhaOutProj weights' g''''
  where
    reshape' ::
      forall batchDim seqDim headDim headEmbedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
    reshape' [batchDim, seqDim, headDim, headEmbedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]))
         )
        (reshape @( 'Shape '[batchDim, seqDim, headDim, headEmbedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, headDim, headEmbedDim]
        input
    reshape'' ::
      forall batchDim seqDim embedDim requiresGradient layout device dataType shape.
      WithShapeC
        ( 'Shape '[batchDim, seqDim, embedDim])
        ( Tensor requiresGradient layout device dataType shape ->
          Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
        ) =>
      [Dim String Integer] ->
      Tensor requiresGradient layout device dataType shape ->
      Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
    reshape'' [batchDim, seqDim, embedDim] input =
      withoutShape
        @( 'Shape '[batchDim, seqDim, embedDim])
        @( Tensor requiresGradient layout device dataType shape ->
           Tensor requiresGradient layout device dataType (ReshapeF shape ( 'Shape '[batchDim, seqDim, embedDim]))
         )
        (reshape @( 'Shape '[batchDim, seqDim, embedDim]) @requiresGradient @layout @device @dataType @shape)
        [batchDim, seqDim, embedDim]
        input

type TestDevice :: Device (DeviceType Nat)
type TestDevice = 'Device 'CPU
type TestLayout = 'Layout 'Dense
type TestDataType = 'DataType 'Float
-- type TestEmbedDim = 'Dim ('Name "embed") ('Size 768)
type TestEmbedDim = 'Dim ('Name "*") ('Size 768)
-- type TestFFNDim = 'Dim ('Name "ffn") ('Size 256)
type TestFFNDim = 'Dim ('Name "*") ('Size 256)
-- type TestQueryEmbedDim = 'Dim ('Name "queryEmbed") ('Size 512)
type TestQueryEmbedDim = 'Dim ('Name "*") ('Size 512)
-- type TestKeyEmbedDim = 'Dim ('Name "keyEmbed") ('Size 2048)
type TestKeyEmbedDim = 'Dim ('Name "*") ('Size 2048)
-- type TestValueEmbedDim = 'Dim ('Name "valueEmbed") ('Size 1024)
type TestValueEmbedDim = 'Dim ('Name "*") ('Size 1024)
-- type TestQuerySeqDim = 'Dim ('Name "querySeq") ('Size 32)
type TestQuerySeqDim = 'Dim ('Name "*") ('Size 32)
-- type TestKeySeqDim = 'Dim ('Name "keySeq") ('Size 48)
type TestKeySeqDim = 'Dim ('Name "*") ('Size 48)
-- type TestBatchDim = 'Dim ('Name "batch") ('Size 4)
type TestBatchDim = 'Dim ('Name "*") ('Size 4)
type TestHeadDim = 'Dim ('Name "head") ('Size 12)
-- type TestHeadDim = 'Dim ('Name "*") ('Size 12)
type TestHeadEmbedDim = 'Dim ('Name "headEmbed") ('Size 64)
-- type TestHeadEmbedDim = 'Dim ('Name "*") ('Size 64)
-- type TestHeadDim :: Dim (Name Symbol) (Size Nat)
-- type TestHeadDim = 'Dim ('Name "head") 'UncheckedSize
-- type TestHeadEmbedDim :: Dim (Name Symbol) (Size Nat)
-- type TestHeadEmbedDim = 'Dim ('Name "headEmbed") 'UncheckedSize

testmha ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
        -- 'UncheckedShape
    )
testmha = do
  g <- mkGenerator @TestDevice 0
  let (result, _) = runState
        ( do
            mha <- state $ initialize @(MultiheadAttention TestDevice TestDataType TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0
            query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
            key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
            value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
            state $ multiheadAttention @TestHeadDim @TestHeadEmbedDim mha query key value -- 12 64
        )
        g
  pure result

data
  TransformerMLP
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerMLP ::
    forall device dataType embedDim ffnDim dropoutP.
    { -- | first fully connected layer
      tmlpLinear0 :: Linear device dataType embedDim ffnDim,
      -- | second fully connected layer
      tmlpLinear1 :: Linear device dataType ffnDim embedDim,
      -- | relu dropout
      tmlpDropout0 :: Dropout dropoutP,
      -- | other dropout
      tmlpDropout1 :: Dropout dropoutP,
      -- | layer norm
      tmlpLayerNorm :: LayerNorm device dataType ('Shape '[embedDim])
    } ->
    TransformerMLP device dataType embedDim ffnDim dropoutP

type HasInitializeTransformerMLPC device dataType embedDim ffnDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)))),
    WithDimC embedDim (WithDimF ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device))),
    WithDimC ffnDim (dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)),
    HasInitializeLinearC device dataType embedDim ffnDim,
    HasInitializeLinearC device dataType ffnDim embedDim,
    HasInitializeLayerNormC device dataType ( 'Shape '[embedDim]),
    Scalar dropoutP
  )

instance
  HasInitializeTransformerMLPC device dataType embedDim ffnDim dropoutP =>
  HasInitialize (TransformerMLP device dataType embedDim ffnDim dropoutP)
  where
  type
    InitializeF (TransformerMLP device dataType embedDim ffnDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    ffnDim
                    ( dropoutP ->
                      Double ->
                      Generator device ->
                      ( TransformerMLP device dataType embedDim ffnDim dropoutP,
                        Generator device
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
            withDim @embedDim $
              \embedDim ->
                withDim @ffnDim @(dropoutP -> Double -> Generator device -> (TransformerMLP device dataType embedDim ffnDim dropoutP, Generator device)) $
                  \ffnDim ->
                    go deviceType dType embedDim ffnDim
    where
      go deviceType dType embedDim ffnDim dropoutP eps = runState $ do
        linear0 <-
          state $
            withoutDim @ffnDim
              ( withoutDim @embedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType embedDim ffnDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  embedDim
              )
              ffnDim
        linear1 <-
          state $
            withoutDim @embedDim
              ( withoutDim @ffnDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(Linear device dataType ffnDim embedDim)
                          )
                          deviceType
                      )
                      dType
                  )
                  ffnDim
              )
              embedDim
        dropout0 <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        dropout1 <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        layerNorm <-
          pure $
            withoutShape @( 'Shape '[embedDim])
              ( withoutDataType @dataType
                  ( withoutDevice @device
                      ( initialize @(LayerNorm device dataType ( 'Shape '[embedDim]))
                      )
                      deviceType
                  )
                  dType
              )
              [embedDim]
              eps
        pure $ TransformerMLP linear0 linear1 dropout0 dropout1 layerNorm

type TransformerMLPOutputLayout inputLayout =
  UnifyLayoutF
    ( UnifyLayoutF
        ( UnifyLayoutF
            inputLayout
            ( UnifyLayoutF
                ( UnifyLayoutF
                    ( UnifyLayoutF
                        (UnifyLayoutF inputLayout ( 'Layout 'Dense))
                        ( 'Layout 'Dense)
                    )
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
        )
        ( 'Layout 'Dense)
    )
    ( 'Layout 'Dense)

type TransformerMLPOutputDevice device inputDevice outputGeneratorDevice =
  -- UnifyDeviceF device (UnifyDeviceF inputDevice outputGeneratorDevice)
  UnifyDeviceF
    ( UnifyDeviceF
        (UnifyDeviceF inputDevice outputGeneratorDevice)
        device
    )
    device

type TransformerMLPOutputDataType dataType inputDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            inputDataType
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    (UnifyDataTypeF (UnifyDataTypeF inputDataType dataType) dataType)
                    dataType
                )
                dataType
            )
        )
        dataType
    )
    dataType

type TransformerMLPOutputShape embedDim ffnDim inputShape =
  LayerNormF
    ( 'Shape '[embedDim])
    ( 'Shape '[embedDim])
    ( BroadcastShapesF
        inputShape
        ( LinearF
            ( 'Shape '[embedDim, ffnDim])
            ( 'Shape '[embedDim])
            ( LinearF
                ( 'Shape '[ffnDim, embedDim])
                ( 'Shape '[ffnDim])
                inputShape
            )
        )
    )

type TransformerMLPOutputGeneratorDevice device inputDevice generatorDevice =
  -- UnifyDeviceF generatorDevice (UnifyDeviceF inputDevice device)
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                (UnifyDeviceF (UnifyDeviceF inputDevice device) device)
                generatorDevice
            )
            device
        )
        device
    )
    ( UnifyDeviceF
        (UnifyDeviceF (UnifyDeviceF inputDevice device) device)
        generatorDevice
    )

transformerMLP ::
  forall device dataType embedDim ffnDim dropoutP requiresGradient inputLayout inputDevice inputDataType inputShape generatorDevice outputLayout outputDevice outputDataType outputShape outputGeneratorDevice.
  ( Scalar dropoutP,
    KnownDim embedDim,
    outputLayout ~ TransformerMLPOutputLayout inputLayout,
    outputDevice ~ TransformerMLPOutputDevice device inputDevice outputGeneratorDevice,
    outputDataType ~ TransformerMLPOutputDataType dataType inputDataType,
    outputShape ~ TransformerMLPOutputShape embedDim ffnDim inputShape,
    outputGeneratorDevice ~ TransformerMLPOutputGeneratorDevice device inputDevice generatorDevice
  ) =>
  TransformerMLP device dataType embedDim ffnDim dropoutP ->
  Tensor requiresGradient inputLayout inputDevice inputDataType inputShape ->
  Generator generatorDevice ->
  ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
    Generator outputGeneratorDevice
  )
transformerMLP TransformerMLP {..} =
  let residual f f' x g =
        let (x', g') = f x g
         in f' (x `add` x') g'
      f x g =
        let (x', g') = forward tmlpLinear0 x g
            (x'', g'') = forward tmlpDropout0 (relu x') g'
            (x''', g''') = forward tmlpLinear1 x'' g''
         in forward tmlpDropout1 x''' g'''
   in residual f (forward tmlpLayerNorm)

testmlp ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestEmbedDim])
    )
testmlp = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              mlp <- state $ initialize @(TransformerMLP TestDevice TestDataType TestEmbedDim TestFFNDim Float) 0.0 1e-5
              x <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestEmbedDim])
              state $ transformerMLP mlp x
          )
          g
  pure result

data
  TransformerLayer
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (keyEmbedDim :: Dim (Name Symbol) (Size Nat))
    (valueEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerLayer ::
    forall device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP.
    { -- | multi-head attention
      tlMultiheadAttention :: MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP,
      -- | dropout
      tlAttentionDropout :: Dropout dropoutP,
      -- | layer norm
      tlLayerNorm :: LayerNorm device dataType ('Shape '[queryEmbedDim]),
      -- | MLP
      tlTransformerMLP :: TransformerMLP device dataType queryEmbedDim ffnDim dropoutP
    } ->
    TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP

type HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))))),
    WithDimC embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))))),
    WithDimC ffnDim (WithDimF queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))))),
    WithDimC queryEmbedDim (WithDimF keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)))),
    WithDimC keyEmbedDim (WithDimF valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))),
    WithDimC valueEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)),
    HasInitializeMultiheadAttentionC device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP,
    HasInitializeLayerNormC device dataType ( 'Shape '[queryEmbedDim]),
    HasInitializeTransformerMLPC device dataType queryEmbedDim ffnDim dropoutP,
    Scalar dropoutP
  )

instance
  HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP =>
  HasInitialize (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
  where
  type
    InitializeF (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP) =
      WithDeviceF
        device
        ( WithDataTypeF
            dataType
            ( WithDimF
                embedDim
                ( WithDimF
                    ffnDim
                    ( WithDimF
                        queryEmbedDim
                        ( WithDimF
                            keyEmbedDim
                            ( WithDimF
                                valueEmbedDim
                                (dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device))
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
            withDim @embedDim $
              \embedDim ->
                withDim @ffnDim $
                  \ffnDim ->
                    withDim @queryEmbedDim $
                      \queryEmbedDim ->
                        withDim @keyEmbedDim $
                          \keyEmbedDim ->
                            withDim @valueEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP, Generator device)) $
                              \valueEmbedDim ->
                                go deviceType dType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim
    where
      go deviceType dType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP eps = runState $ do
        multiheadAttention <-
          state $
            withoutDim @valueEmbedDim
              ( withoutDim @keyEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @embedDim
                          ( withoutDataType @dataType
                              ( withoutDevice @device
                                  ( initialize @(MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP)
                                  )
                                  deviceType
                              )
                              dType
                          )
                          embedDim
                      )
                      queryEmbedDim
                  )
                  keyEmbedDim
              )
              valueEmbedDim
              dropoutP
        attentionDropout <-
          pure $ initialize @(Dropout dropoutP) dropoutP
        layerNorm <-
          pure $
            withoutShape @( 'Shape '[queryEmbedDim])
              ( withoutDataType @dataType
                  ( withoutDevice @device
                      ( initialize @(LayerNorm device dataType ( 'Shape '[queryEmbedDim]))
                      )
                      deviceType
                  )
                  dType
              )
              [queryEmbedDim]
              eps
        transformerMLP <-
          state $
            withoutDim @ffnDim
              ( withoutDim @queryEmbedDim
                  ( withoutDataType @dataType
                      ( withoutDevice @device
                          ( initialize @(TransformerMLP device dataType queryEmbedDim ffnDim dropoutP)
                          )
                          deviceType
                      )
                      dType
                  )
                  queryEmbedDim
              )
              ffnDim
              dropoutP
              eps
        pure $ TransformerLayer multiheadAttention attentionDropout layerNorm transformerMLP

type TransformerLayerOutputLayout queryLayout multiheadAttentionOutputLayout =
  UnifyLayoutF
    ( UnifyLayoutF
        ( UnifyLayoutF
            ( UnifyLayoutF
                ( UnifyLayoutF
                    (UnifyLayoutF queryLayout multiheadAttentionOutputLayout)
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
            ( UnifyLayoutF
                ( UnifyLayoutF
                    ( UnifyLayoutF
                        ( UnifyLayoutF
                            ( UnifyLayoutF
                                ( UnifyLayoutF
                                    ( UnifyLayoutF
                                        queryLayout
                                        multiheadAttentionOutputLayout
                                    )
                                    ( 'Layout 'Dense)
                                )
                                ( 'Layout 'Dense)
                            )
                            ( 'Layout 'Dense)
                        )
                        ( 'Layout 'Dense)
                    )
                    ( 'Layout 'Dense)
                )
                ( 'Layout 'Dense)
            )
        )
        ( 'Layout 'Dense)
    )
    ( 'Layout 'Dense)

type TransformerLayerOutputLayout' queryLayout keyLayout valueLayout = TransformerLayerOutputLayout queryLayout (MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout)

type TransformerLayerOutputDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice outputGeneratorDevice =
  -- UnifyDeviceF outputGeneratorDevice (UnifyDeviceF multiheadAttentionOutputGeneratorDevice (UnifyDeviceF multiheadAttentionOutputDevice (UnifyDeviceF queryDevice device)))
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        queryDevice
                        ( UnifyDeviceF
                            multiheadAttentionOutputDevice
                            multiheadAttentionOutputGeneratorDevice
                        )
                    )
                    device
                )
                device
            )
            outputGeneratorDevice
        )
        device
    )
    device

type TransformerLayerOutputDevice' device queryDevice keyDevice valueDevice generatorDevice = TransformerLayerOutputDevice device queryDevice (MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice) (MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice) (TransformerLayerOutputGeneratorDevice' device queryDevice keyDevice valueDevice generatorDevice)

type TransformerLayerOutputGeneratorDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice =
  -- UnifyDeviceF multiheadAttentionOutputGeneratorDevice (UnifyDeviceF multiheadAttentionOutputDevice (UnifyDeviceF queryDevice device))
  UnifyDeviceF
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        ( UnifyDeviceF
                            ( UnifyDeviceF
                                ( UnifyDeviceF
                                    queryDevice
                                    ( UnifyDeviceF
                                        multiheadAttentionOutputDevice
                                        multiheadAttentionOutputGeneratorDevice
                                    )
                                )
                                device
                            )
                            device
                        )
                        device
                    )
                    device
                )
                ( UnifyDeviceF
                    multiheadAttentionOutputDevice
                    multiheadAttentionOutputGeneratorDevice
                )
            )
            device
        )
        device
    )
    ( UnifyDeviceF
        ( UnifyDeviceF
            ( UnifyDeviceF
                ( UnifyDeviceF
                    ( UnifyDeviceF
                        ( UnifyDeviceF
                            queryDevice
                            ( UnifyDeviceF
                                multiheadAttentionOutputDevice
                                multiheadAttentionOutputGeneratorDevice
                            )
                        )
                        device
                    )
                    device
                )
                device
            )
            device
        )
        ( UnifyDeviceF
            multiheadAttentionOutputDevice
            multiheadAttentionOutputGeneratorDevice
        )
    )

type TransformerLayerOutputGeneratorDevice' device queryDevice keyDevice valueDevice generatorDevice = TransformerLayerOutputGeneratorDevice device queryDevice (MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice) (MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice)

type TransformerLayerOutputDataType dataType queryDataType multiheadAttentionOutputDataType =
  UnifyDataTypeF
    ( UnifyDataTypeF
        ( UnifyDataTypeF
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    (UnifyDataTypeF queryDataType multiheadAttentionOutputDataType)
                    dataType
                )
                dataType
            )
            ( UnifyDataTypeF
                ( UnifyDataTypeF
                    ( UnifyDataTypeF
                        ( UnifyDataTypeF
                            ( UnifyDataTypeF
                                ( UnifyDataTypeF
                                    ( UnifyDataTypeF
                                        queryDataType
                                        multiheadAttentionOutputDataType
                                    )
                                    dataType
                                )
                                dataType
                            )
                            dataType
                        )
                        dataType
                    )
                    dataType
                )
                dataType
            )
        )
        dataType
    )
    dataType

type TransformerLayerOutputDataType' dataType queryDataType keyDataType valueDataType = TransformerLayerOutputDataType dataType queryDataType (MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType)

type TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape =
  LayerNormF
    ( 'Shape '[queryEmbedDim])
    ( 'Shape '[queryEmbedDim])
    ( BroadcastShapesF
        ( LayerNormF
            ( 'Shape '[queryEmbedDim])
            ( 'Shape '[queryEmbedDim])
            (BroadcastShapesF queryShape multiheadAttentionOutputShape)
        )
        ( LinearF
            ( 'Shape '[queryEmbedDim, ffnDim])
            ( 'Shape '[queryEmbedDim])
            ( LinearF
                ( 'Shape '[ffnDim, queryEmbedDim])
                ( 'Shape '[ffnDim])
                ( LayerNormF
                    ( 'Shape '[queryEmbedDim])
                    ( 'Shape '[queryEmbedDim])
                    (BroadcastShapesF queryShape multiheadAttentionOutputShape)
                )
            )
        )
    )

type TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim queryShape keyShape valueShape = TransformerLayerOutputShape ffnDim queryEmbedDim queryShape (MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim (BatchDim queryShape keyShape valueShape) (QuerySeqDim queryShape) (KeySeqDim keyShape valueShape) queryShape keyShape valueShape)

transformerLayer ::
  forall headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputDataType outputShape outputGeneratorDevice multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape.
  ( MultiheadAttentionC headDim headEmbedDim batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape,
    WithDimC headDim (WithDimF headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient outputLayout outputDevice outputDataType outputShape, Generator outputGeneratorDevice))),
    WithDimC headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient outputLayout outputDevice outputDataType outputShape, Generator outputGeneratorDevice)),
    multiheadAttentionOutputLayout ~ MultiheadAttentionOutputLayout queryLayout keyLayout valueLayout,
    multiheadAttentionOutputDevice ~ MultiheadAttentionOutputDevice device queryDevice keyDevice valueDevice generatorDevice,
    multiheadAttentionOutputGeneratorDevice ~ MultiheadAttentionOutputGeneratorDevice device queryDevice keyDevice generatorDevice,
    multiheadAttentionOutputDataType ~ MultiheadAttentionOutputDataType dataType queryDataType keyDataType valueDataType,
    multiheadAttentionOutputShape ~ MultiheadAttentionOutputShape embedDim queryEmbedDim keyEmbedDim valueEmbedDim headDim headEmbedDim batchDim querySeqDim keySeqDim queryShape keyShape valueShape,
    KnownDim queryEmbedDim,
    outputShape ~ TransformerLayerOutputShape ffnDim queryEmbedDim queryShape multiheadAttentionOutputShape,
    outputDataType ~ TransformerLayerOutputDataType dataType queryDataType multiheadAttentionOutputDataType,
    outputLayout ~ TransformerLayerOutputLayout queryLayout multiheadAttentionOutputLayout,
    outputDevice ~ TransformerLayerOutputDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice outputGeneratorDevice,
    outputGeneratorDevice ~ TransformerLayerOutputGeneratorDevice device queryDevice multiheadAttentionOutputDevice multiheadAttentionOutputGeneratorDevice
  ) =>
  -- | transformer layer model
  TransformerLayer device dataType embedDim ffnDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  WithDimF
    headDim
    ( WithDimF
        headEmbedDim
        ( Generator generatorDevice ->
          ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
            Generator outputGeneratorDevice
          )
        )
    )
transformerLayer TransformerLayer {..} query key value =
  withDim @headDim $ \headDim ->
    withDim @headEmbedDim $ \headEmbedDim g ->
      let residual f f' query g =
            let (query', g') = f query g
             in f' (query `add` query') g'
          f (query :: Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (g :: Generator generatorDevice) =
            let (query' :: Tensor requiresGradient multiheadAttentionOutputLayout multiheadAttentionOutputDevice multiheadAttentionOutputDataType multiheadAttentionOutputShape, g' :: (Generator multiheadAttentionOutputGeneratorDevice)) =
                  withoutDim @headEmbedDim
                    ( withoutDim @headDim
                        ( multiheadAttention @headDim @headEmbedDim @generatorDevice tlMultiheadAttention query key value
                        )
                        headDim
                    )
                    headEmbedDim
                    g
             in forward tlAttentionDropout query' g'
          (query', g') = residual f (forward tlLayerNorm) query g
       in transformerMLP tlTransformerMLP query' g'

testtl ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
    )
testtl = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              tl <- state $ initialize @(TransformerLayer TestDevice TestDataType TestEmbedDim TestFFNDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0 1e-5
              query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
              key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
              value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
              state $ transformerLayer @TestHeadDim @TestHeadEmbedDim tl query key value
          )
          g
  pure result

data
  TransformerLayerStack
    (numLayers :: Nat)
    (device :: Device (DeviceType Nat))
    (dataType :: DataType DType)
    (headDim :: Dim (Name Symbol) (Size Nat))
    (headEmbedDim :: Dim (Name Symbol) (Size Nat))
    (embedDim :: Dim (Name Symbol) (Size Nat))
    (ffnDim :: Dim (Name Symbol) (Size Nat))
    (queryEmbedDim :: Dim (Name Symbol) (Size Nat))
    (dropoutP :: Type)
  where
  TransformerLayerStackNil ::
    TransformerLayerStack 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  TransformerLayerStackCons ::
    -- | head dim
    Dim String Integer ->
    -- | head embed dim
    Dim String Integer ->
    TransformerLayer device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP ->
    TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP ->
    TransformerLayerStack (numLayers + 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP

class HasInitializeTransformerLayerStack (nil :: Bool) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP where
  initializeTransformerLayerStack ::
    Proxy nil ->
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
                          ffnDim
                          ( WithDimF
                              queryEmbedDim
                              (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
                          )
                      )
                  )
              )
          )
      )

type HasInitializeTransformerLayerStackC numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP =
  ( WithDeviceC device (WithDataTypeF dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))))))),
    WithDataTypeC dataType (WithDimF headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))))))),
    WithDimC headDim (WithDimF headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))))),
    WithDimC headEmbedDim (WithDimF embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))))),
    WithDimC embedDim (WithDimF ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)))),
    WithDimC ffnDim (WithDimF queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))),
    WithDimC queryEmbedDim (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
  )

instance
  HasInitializeTransformerLayerStackC 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP =>
  HasInitializeTransformerLayerStack 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  where
  initializeTransformerLayerStack _ =
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
                        withDim @ffnDim $
                          \_ffnDim ->
                            withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)) $
                              \_queryEmbedDim _dropoutP _eps g -> (TransformerLayerStackNil, g)

instance
  ( HasInitializeTransformerLayerC device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP,
    HasInitializeTransformerLayerStackC (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP,
    HasInitializeTransformerLayerStackC numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP,
    HasInitialize (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
  ) =>
  HasInitializeTransformerLayerStack 'True numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP
  where
  initializeTransformerLayerStack _ =
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
                        withDim @ffnDim $
                          \ffnDim ->
                            withDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device)) $
                              \queryEmbedDim ->
                                go deviceType dType headDim headEmbedDim embedDim ffnDim queryEmbedDim
    where
      go deviceType dType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP eps = runState $ do
        layerStack <-
          state $
            withoutDim @queryEmbedDim @(dropoutP -> Double -> Generator device -> (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
              ( withoutDim @ffnDim
                  ( withoutDim @embedDim
                      ( withoutDim @headEmbedDim
                          ( withoutDim @headDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
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
                  ffnDim
              )
              queryEmbedDim
              dropoutP
              eps
        layer <-
          state $
            withoutDim @queryEmbedDim
              ( withoutDim @queryEmbedDim
                  ( withoutDim @queryEmbedDim
                      ( withoutDim @ffnDim
                          ( withoutDim @embedDim
                              ( withoutDataType @dataType
                                  ( withoutDevice @device
                                      ( initialize @(TransformerLayer device dataType embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP)
                                      )
                                      deviceType
                                  )
                                  dType
                              )
                              embedDim
                          )
                          ffnDim
                      )
                      queryEmbedDim
                  )
                  queryEmbedDim
              )
              queryEmbedDim
              dropoutP
              eps
        pure $ TransformerLayerStackCons headDim headEmbedDim layer layerStack

instance
  HasInitializeTransformerLayerStack (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP =>
  HasInitialize (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP)
  where
  type
    InitializeF (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) =
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
                            ffnDim
                            ( WithDimF
                                queryEmbedDim
                                (dropoutP -> Double -> Generator device -> (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP, Generator device))
                            )
                        )
                    )
                )
            )
        )
  initialize = initializeTransformerLayerStack @(1 <=? numLayers) @numLayers @device @dataType @headDim @headEmbedDim @embedDim @ffnDim @queryEmbedDim @dropoutP (Proxy :: Proxy (1 <=? numLayers))

class HasForwardTransformerLayerStack (nil :: Bool) (numLayers :: Nat) (headDim :: Dim (Name Symbol) (Size Nat)) (headEmbedDim :: Dim (Name Symbol) (Size Nat)) device dataType (embedDim :: Dim (Name Symbol) (Size Nat)) (ffnDim :: Dim (Name Symbol) (Size Nat)) queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType (queryShape :: Shape [Dim (Name Symbol) (Size Nat)]) (generatorDevice :: Device (DeviceType Nat)) where
  type HasForwardTransformerLayerStackOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice :: Type
  type HasForwardTransformerLayerStackGeneratorOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice :: Type
  forwardTransformerLayerStack ::
    Proxy nil ->
    TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP ->
    Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
    Generator generatorDevice ->
    (HasForwardTransformerLayerStackOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice,
     HasForwardTransformerLayerStackGeneratorOutput nil numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice)

instance
  ( WithDimC headDim (WithDimF headEmbedDim (HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice)),
    WithDimC headEmbedDim (HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice)
  ) =>
  HasForwardTransformerLayerStack 'False 0 headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice
  where
  type HasForwardTransformerLayerStackOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice = Tensor requiresGradient queryLayout queryDevice queryDataType queryShape
  type HasForwardTransformerLayerStackGeneratorOutput 'False 0 device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice = Generator generatorDevice
  forwardTransformerLayerStack _ TransformerLayerStackNil x g = (x, g)

instance
  ( MultiheadAttentionC headDim headEmbedDim (BatchDim queryShape queryShape queryShape) (QuerySeqDim queryShape) (QuerySeqDim queryShape) device dataType embedDim queryEmbedDim queryEmbedDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape queryLayout queryDevice queryDataType queryShape queryLayout queryDevice queryDataType queryShape generatorDevice (MultiheadAttentionOutputLayout queryLayout queryLayout queryLayout) (MultiheadAttentionOutputDevice device queryDevice queryDevice queryDevice generatorDevice) (MultiheadAttentionOutputGeneratorDevice device queryDevice queryDevice generatorDevice) (MultiheadAttentionOutputDataType dataType queryDataType queryDataType queryDataType) (MultiheadAttentionOutputShape embedDim queryEmbedDim queryEmbedDim queryEmbedDim headDim headEmbedDim (BatchDim queryShape queryShape queryShape) (QuerySeqDim queryShape) (QuerySeqDim queryShape) queryShape queryShape queryShape),
    WithDimC headDim (WithDimF headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape), Generator (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice)))),
    WithDimC headEmbedDim (Generator generatorDevice -> (Tensor requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape), Generator (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice))),
    KnownDim queryEmbedDim,
    HasForward (TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape)) (Generator (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice))
  ) =>
  HasForwardTransformerLayerStack 'True numLayers headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice
  where
  type HasForwardTransformerLayerStackOutput 'True numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice = HasForwardTransformerLayerStackOutput (1 <=? numLayers - 1) (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape) (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice)
  type HasForwardTransformerLayerStackGeneratorOutput 'True numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice = HasForwardTransformerLayerStackGeneratorOutput (1 <=? numLayers - 1) (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape) (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice)
  forwardTransformerLayerStack _ (TransformerLayerStackCons headDim headEmbedDim layer layerStack) x g =
    let (x', g') =
          withoutDim @headEmbedDim
            ( withoutDim @headDim
                ( transformerLayer @headDim @headEmbedDim @(BatchDim queryShape queryShape queryShape) @(QuerySeqDim queryShape) @(QuerySeqDim queryShape) @device @dataType @embedDim @ffnDim @queryEmbedDim @queryEmbedDim @queryEmbedDim @dropoutP @requiresGradient @queryLayout @queryDevice @queryDataType @queryShape @queryLayout @queryDevice @queryDataType @queryShape @queryLayout @queryDevice @queryDataType @queryShape @generatorDevice layer x x x
                )
                headDim
            )
            headEmbedDim
            g
     in forward @(TransformerLayerStack (numLayers - 1) device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) @(Tensor requiresGradient (TransformerLayerOutputLayout' queryLayout queryLayout queryLayout) (TransformerLayerOutputDevice' device queryDevice queryDevice queryDevice generatorDevice) (TransformerLayerOutputDataType' dataType queryDataType queryDataType queryDataType) (TransformerLayerOutputShape' headDim headEmbedDim embedDim ffnDim queryEmbedDim queryEmbedDim queryEmbedDim queryShape queryShape queryShape)) @(Generator (TransformerLayerOutputGeneratorDevice' device queryDevice queryDevice queryDevice generatorDevice)) layerStack x' g'

instance
  ( HasForwardTransformerLayerStack (1 <=? numLayers) numLayers headDim headEmbedDim device dataType embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice
  ) =>
  HasForward (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (Generator generatorDevice)
  where
  type
    ForwardOutput (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (Generator generatorDevice) =
      HasForwardTransformerLayerStackOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice
  type
    ForwardGeneratorOutput (TransformerLayerStack numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP) (Tensor requiresGradient queryLayout queryDevice queryDataType queryShape) (Generator generatorDevice) =
      HasForwardTransformerLayerStackGeneratorOutput (1 <=? numLayers) numLayers device dataType headDim headEmbedDim embedDim ffnDim queryEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape generatorDevice
     
  forward = forwardTransformerLayerStack (Proxy @(1 <=? numLayers))

testtlstack ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape _)
    )
testtlstack = do
  g <- mkGenerator @TestDevice 0
  let (result, _) =
        runState
          ( do
              tlstack <- state $ initialize @(TransformerLayerStack 2 TestDevice TestDataType TestHeadDim TestHeadEmbedDim TestEmbedDim TestFFNDim TestQueryEmbedDim Float) 0.0 1e-5
              x <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[ TestBatchDim, TestQuerySeqDim, 'Dim 'UncheckedName 'UncheckedSize]) (Dim "*" 512)
              state $ forward tlstack x
          )
          g
  pure result