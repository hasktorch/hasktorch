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

module Torch.GraduallyTyped.NN.Transformer where

import Control.Monad.State.Strict (MonadState (state), runState)
import Data.Kind (Type)
import GHC.TypeLits (type (<=), type (*), KnownNat, Div, Nat, Symbol)
import Torch.DType (DType (..))
import Torch.GraduallyTyped.DType (UnifyDataTypeF, DataType (DataType), WithDataTypeC (..))
import Torch.GraduallyTyped.Device (UnifyDeviceF, Device (..), DeviceType(..), WithDeviceC (..))
import Torch.GraduallyTyped.NN.Class (forward, HasInitialize (..))
import Torch.GraduallyTyped.NN.Dropout (Dropout (Dropout))
import Torch.GraduallyTyped.NN.Linear (HasInitializeLinearC, Linear (Linear))
import Torch.GraduallyTyped.Random (generator, Generator)
import Torch.GraduallyTyped.Scalar (Scalar)
import Torch.GraduallyTyped.Shape (UnifyDimF, GetDimF, dimSize, Size(..), Name(..), KnownDim(..), WithShapeC(..), NumelF, By(..), SelectDim(..), Dim (..), Shape (..), WithDimC (..))
import Torch.GraduallyTyped.Tensor.MathOperations.Pointwise (divScalar, add)
import Torch.GraduallyTyped.Tensor.Type (Tensor)
import Torch.GraduallyTyped.Tensor.IndexingSlicingJoining (ReshapeF, TransposeF, reshape, transpose)
import Torch.GraduallyTyped.Tensor.MathOperations.BlasLapack (MatmulF, matmul)
import Torch.GraduallyTyped.NN.Functional.Linear (LinearF)
import Torch.GraduallyTyped.NN.Functional.NonLinearActivation (SoftmaxF, softmax)
import Torch.GraduallyTyped.Layout
import Torch.GraduallyTyped.RequiresGradient (RequiresGradient(Dependent))
import Torch.GraduallyTyped.Tensor.Creation (randn)

--------------------------------------------------------------------------------
-- Multi-Headed Attention Layer
--------------------------------------------------------------------------------

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

-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) queryShape
-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) keyShape
-- batchDim ~ GetDimF ('SelectDim ('ByIndex 0)) valueShape
-- querySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) queryShape
-- keySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) keyShape
-- keySeqDim ~ GetDimF ('SelectDim ('ByIndex 1)) valueShape
-- queryEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) queryShape
-- keyEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) keyShape
-- valueEmbedDim ~ GetDimF ('SelectDim ('ByIndex 2)) valueShape
-- headDim
-- headEmbedDim
-- embedDim
multiheadAttention ::
  forall (headDim :: Dim (Name Symbol) (Size Nat)) (headEmbedDim :: Dim (Name Symbol) (Size Nat)) batchDim querySeqDim keySeqDim device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP requiresGradient queryLayout queryDevice queryDataType queryShape keyLayout keyDevice keyDataType keyShape valueLayout valueDevice valueDataType valueShape generatorDevice outputLayout outputDevice outputDataType outputShape.
  ( KnownDim embedDim,
    KnownDim headEmbedDim,
    KnownDim headDim,
    KnownDim keySeqDim,
    KnownDim querySeqDim,
    KnownDim batchDim,
    Scalar dropoutP,
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
              outputDevice
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
              outputDevice
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          ( UnifyDataTypeF
              ( UnifyDataTypeF
                  (UnifyDataTypeF (UnifyDataTypeF queryDataType dataType) dataType)
                  (UnifyDataTypeF (UnifyDataTypeF keyDataType dataType) dataType)
              )
              (UnifyDataTypeF (UnifyDataTypeF valueDataType dataType) dataType)
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
                              ( 'Shape '[batchDim, keySeqDim, headDim, headEmbedDim])
                          )
                      )
                  )
              )
              ( 'Shape '[batchDim, querySeqDim, embedDim])
          )
      ),
    UnifyDeviceF
      ( UnifyDeviceF
          (UnifyDeviceF (UnifyDeviceF queryDevice device) device)
          (UnifyDeviceF (UnifyDeviceF keyDevice device) device)
      )
      generatorDevice
      ~ outputDevice,
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
      ~ outputShape,
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
      ~ outputDataType,
    UnifyDeviceF
      ( UnifyDeviceF
          ( UnifyDeviceF
              outputDevice
              (UnifyDeviceF (UnifyDeviceF valueDevice device) device)
          )
          device
      )
      device
      ~ outputDevice,
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
      ~ outputLayout,
    batchDim ~ UnifyDimF (UnifyDimF (GetDimF ( 'SelectDim ( 'ByIndex 0)) queryShape) (GetDimF ( 'SelectDim ( 'ByIndex 0)) keyShape)) (GetDimF ( 'SelectDim ( 'ByIndex 0)) valueShape),
    querySeqDim ~ GetDimF ( 'SelectDim ( 'ByIndex 1)) queryShape,
    keySeqDim ~ UnifyDimF (GetDimF ( 'SelectDim ( 'ByIndex 1)) keyShape) (GetDimF ( 'SelectDim ( 'ByIndex 1)) valueShape)
  ) =>
  -- | multi-head attention model
  MultiheadAttention device dataType embedDim queryEmbedDim keyEmbedDim valueEmbedDim dropoutP ->
  -- | query representation
  Tensor requiresGradient queryLayout queryDevice queryDataType queryShape ->
  -- | key representation
  Tensor requiresGradient keyLayout keyDevice keyDataType keyShape ->
  -- | value representation
  Tensor requiresGradient valueLayout valueDevice valueDataType valueShape ->
  -- | random generator
  Generator generatorDevice ->
  -- | attention and random generator
  ( Tensor requiresGradient outputLayout outputDevice outputDataType outputShape,
    Generator outputDevice
  )
multiheadAttention MultiheadAttention {..} query key value g =
  let batchDim = case dimVal @batchDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      querySeqDim = case dimVal @querySeqDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      keySeqDim = case dimVal @keySeqDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      headDim = case dimVal @headDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      headEmbedDim = case dimVal @headEmbedDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      embedDim = case dimVal @embedDim of
        Dim (Name name) (Size size) -> Dim name size
        Dim _ _ -> undefined
      scaling :: Double = sqrt . fromIntegral . dimSize $ headDim
      q =
        transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @querySeqDim @headDim @headEmbedDim [batchDim, querySeqDim, headDim, headEmbedDim]
          . flip divScalar scaling
          . forward @_ @_ @() mhaQInProj
          $ query
      k = transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @keySeqDim @headDim @headEmbedDim [batchDim, keySeqDim, headDim, headEmbedDim]
          . forward @_ @_ @() mhaKInProj $ key
      qk = q `matmul` transpose @( 'SelectDim ( 'ByIndex 2)) @( 'SelectDim ( 'ByIndex 3)) k
      (weights, g') = forward @_ @_ @(Generator generatorDevice) mhaDropout (softmax @( 'SelectDim ( 'ByIndex 3)) qk) g
      v = transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2))
          . reshape' @batchDim @keySeqDim @headDim @headEmbedDim  [batchDim, keySeqDim, headDim, headEmbedDim]
          . forward @_ @_ @() mhaVInProj $ value
   in (forward @_ @_ @() mhaOutProj . reshape'' @batchDim @querySeqDim @embedDim [batchDim, querySeqDim, embedDim] . transpose @( 'SelectDim ( 'ByIndex 1)) @( 'SelectDim ( 'ByIndex 2)) $ weights `matmul` v, g')
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
    reshape''
      :: forall batchDim seqDim embedDim requiresGradient layout device dataType shape.
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
type TestEmbedDim = 'Dim ('Name "embed") ('Size 32)
type TestQuerySeqDim = 'Dim ('Name "querySeq") ('Size 64)
type TestQueryEmbedDim = 'Dim ('Name "queryEmbed") ('Size 32)
type TestKeySeqDim = 'Dim ('Name "keySeq") ('Size 48)
type TestKeyEmbedDim = 'Dim ('Name "keyEmbed") ('Size 16)
type TestValueEmbedDim = 'Dim ('Name "valueEmbed") ('Size 24)
type TestHeadDim = 'Dim ('Name "head") ('Size 8)
type TestHeadEmbedDim = 'Dim ('Name "headEmbed") ('Size 4)
type TestBatchDim = 'Dim ('Name "batch") ('Size 4)

testmha ::
  IO
    ( Tensor
        'Dependent
        ( 'Layout 'Dense)
        ( 'Device 'CPU)
        ( 'DataType 'Float)
        ( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
    )
testmha = do
  g <- generator @TestDevice 0
  let (result, _) =
        runState
          ( do
              mha <- state $ initialize @(MultiheadAttention TestDevice TestDataType TestEmbedDim TestQueryEmbedDim TestKeyEmbedDim TestValueEmbedDim Float) 0.0
              query <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestQuerySeqDim, TestQueryEmbedDim])
              key <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestKeyEmbedDim])
              value <- state $ randn @ 'Dependent @TestLayout @TestDevice @TestDataType @( 'Shape '[TestBatchDim, TestKeySeqDim, TestValueEmbedDim])
              state $ multiheadAttention @TestHeadDim @TestHeadEmbedDim mha query key value
          )
          g
  pure result
