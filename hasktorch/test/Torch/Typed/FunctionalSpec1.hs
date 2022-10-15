{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -freduction-depth=0 #-}

module Torch.Typed.FunctionalSpec1
  ( Torch.Typed.FunctionalSpec1.spec,
  )
where

import Data.Proxy
import GHC.TypeLits
import Test.Hspec (Spec, before_, describe, it)
import Test.QuickCheck ()
import Torch.Internal.Managed.Type.Context (get_manual_seed)
import Torch.Typed
import Torch.Typed.AuxiliarySpec
import Prelude hiding
  ( abs,
    acos,
    acosh,
    all,
    any,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    exp,
    floor,
    log,
    max,
    min,
    round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
  )

data SqueezeAllSpec = SqueezeAllSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ SqueezeAll shape
  ) =>
  Apply' SqueezeAllSpec ((Proxy device, (Proxy dtype, Proxy shape)), IO ()) (IO ())
  where
  apply' SqueezeAllSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = squeezeAll t
      checkDynamicTensorAttributes t'

data TransposeSpec = TransposeSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    shape' ~ Transpose shape n m,
    KnownNat n,
    KnownNat m
  ) =>
  Apply' TransposeSpec (((Proxy n, Proxy m), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' TransposeSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = transpose @n @m t
      checkDynamicTensorAttributes t'

data Transpose2DSpec = Transpose2DSpec

instance
  ( TensorOptions '[i, j] dtype device,
    TensorOptions '[j, i] dtype device
  ) =>
  Apply' Transpose2DSpec ((Proxy device, (Proxy dtype, Proxy '[i, j])), IO ()) (IO ())
  where
  apply' Transpose2DSpec (_, agg) =
    agg >> do
      let t = ones @'[i, j] @dtype @device
          t' = transpose2D t
      checkDynamicTensorAttributes t'

data NarrowSpec = NarrowSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions
      ( NarrowCheck
          (ExtractDim dim shape)
          (Narrow' dim shape (ExtractDim dim shape) start length)
          shape
          dim
          start
          length
      )
      dtype
      device,
    All KnownNat shape,
    All KnownNat '[dim, start, length]
  ) =>
  Apply' NarrowSpec ((Proxy dim, (Proxy start, (Proxy length, (Proxy device, (Proxy dtype, Proxy shape))))), IO ()) (IO ())
  where
  apply' NarrowSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
          t' = narrow @dim @start @length t
      checkDynamicTensorAttributes t'

data DiagSpec = DiagSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownTri tri,
    KnownNat index,
    StandardDTypeValidation device dtype,
    shape' ~ DiagShape tri index shape
  ) =>
  Apply' DiagSpec (((Proxy tri, Proxy index), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      checkDynamicTensorAttributes $ diag @tri @index t

data DiagEmbedSpec = DiagEmbedSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat index,
    KnownNat dim1,
    KnownNat dim2,
    DimsDistinctAscending dim1 dim2,
    shape' ~ DiagEmbedShape index dim1 dim2 shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagEmbedSpec (((Proxy index, (Proxy dim1, Proxy dim2)), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagEmbedSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      foldMap
        (\tri -> checkDynamicTensorAttributes $ diagEmbed @index @dim1 @dim2 tri t)
        [Upper, Lower]

data DiagflatSpec = DiagflatSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownNat index,
    shape' ~ DiagflatShape index shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagflatSpec ((Proxy index, (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagflatSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      foldMap
        (\tri -> checkDynamicTensorAttributes $ diagflat @index tri t)
        [Upper, Lower]

data DiagonalSpec = DiagonalSpec

instance
  ( TensorOptions shape dtype device,
    TensorOptions shape' dtype device,
    KnownTri tri,
    KnownNat index,
    KnownNat dim1,
    KnownNat dim2,
    NDimAtLeast 2 shape,
    DimsDistinctAscending dim1 dim2,
    shape' ~ DiagonalShape tri index dim1 dim2 shape,
    StandardDTypeValidation device dtype
  ) =>
  Apply' DiagonalSpec (((Proxy tri, (Proxy index, (Proxy dim1, Proxy dim2))), (Proxy device, (Proxy dtype, Proxy shape))), IO ()) (IO ())
  where
  apply' DiagonalSpec (_, agg) =
    agg >> do
      let t = ones @shape @dtype @device
      checkDynamicTensorAttributes $ diagonal @tri @index @dim1 @dim2 t

spec :: Spec
spec = before_ printSeed $ do
  foldMap spec' availableDevices
  where
    printSeed = do
      putStr "      seed:"
      get_manual_seed >>= print

spec' :: Device -> Spec
spec' device =
  describe ("for " <> show device) $ do
    let standardShapes = Proxy @'[2, 3] :. HNil -- (Proxy :: Proxy ('[] :: [Nat])) :. Proxy @'[0]  :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. HNil
        squareShapes = Proxy @'[0, 0] :. Proxy @'[1, 1] :. Proxy @'[2, 2] :. Proxy @'[0, 0, 0] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 0] :. Proxy @'[3, 2, 2] :. HNil
        reductions = Proxy @ReduceNone :. Proxy @ReduceMean :. Proxy @ReduceSum :. HNil

    describe "shape ops" $ do
      it "narrow" $ do
        let dims = Proxy @0 :. Proxy @1 :. HNil
            narrowStarts = Proxy @0 :. Proxy @1 :. HNil
            narrowLengths = Proxy @1 :. Proxy @2 :. HNil
            narrowShapes = Proxy @'[3, 3, 2] :. Proxy @'[13, 5, 0] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO NarrowSpec () (hproduct dims (hproduct narrowStarts (hproduct narrowLengths (hattach cpu (hproduct standardFloatingPointDTypes narrowShapes)))))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO NarrowSpec () (hproduct dims (hproduct narrowStarts (hproduct narrowLengths (hattach cuda0 (hproduct allFloatingPointDTypes narrowShapes)))))
      it "squeezeAll" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO SqueezeAllSpec () (hattach cpu (hproduct allDTypes standardShapes))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO SqueezeAllSpec () (hattach cuda0 (hproduct allDTypes standardShapes))
      it "transpose" $ do
        let dims =
              hzip
                (Proxy @0 :. Proxy @0 :. Proxy @1 :. HNil)
                (Proxy @0 :. Proxy @1 :. Proxy @0 :. HNil)
            shapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. Proxy @'[0, 1, 1] :. Proxy @'[1, 0, 1] :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} ->
            hfoldrM @IO TransposeSpec () (hproduct dims (hattach cpu (hproduct allDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} ->
            hfoldrM @IO TransposeSpec () (hproduct dims (hattach cuda0 (hproduct allDTypes shapes)))
      it "transpose2d" $ case device of
        Device {deviceType = CPU, deviceIndex = 0} ->
          hfoldrM @IO Transpose2DSpec () (hattach cpu (hproduct allDTypes (Proxy @'[2, 3] :. HNil)))
        Device {deviceType = CUDA, deviceIndex = 0} ->
          hfoldrM @IO Transpose2DSpec () (hattach cuda0 (hproduct allDTypes (Proxy @'[2, 3] :. HNil)))
      it "diag" $ do
        let vectorShapes = Proxy @'[0] :. Proxy @'[1] :. Proxy @'[2] :. HNil
            emptyShapes = Proxy @'[0, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. HNil
            tris = Proxy @'Upper :. Proxy @'Lower :. HNil
            indexes = Proxy @0 :. Proxy @1 :. HNil
            indexes' = Proxy @0 :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} -> do
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes) (hattach cpu (hproduct standardDTypes standardShapes)))
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes) (hattach cpu (hproduct standardDTypes vectorShapes)))
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes') (hattach cpu (hproduct standardDTypes emptyShapes)))
          Device {deviceType = CUDA, deviceIndex = 0} -> do
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes) (hattach cuda0 (hproduct (withHalf standardDTypes) standardShapes)))
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes) (hattach cuda0 (hproduct (withHalf standardDTypes) vectorShapes)))
            hfoldrM @IO DiagSpec () (hproduct (hproduct tris indexes') (hattach cuda0 (hproduct (withHalf standardDTypes) emptyShapes)))
      it "diagEmbed" $ do
        let shapes =
              standardShapes
                `happend` ( Proxy @'[0]
                              :. Proxy @'[1]
                              :. Proxy @'[2]
                              :. Proxy @'[0, 0]
                              :. Proxy @'[0, 1]
                              :. Proxy @'[1, 0]
                              :. HNil
                          )
            indexes = Proxy @0 :. Proxy @1 :. HNil
            dims = (Proxy @0, Proxy @1) :. HNil
            allDims = (Proxy @0, Proxy @2) :. dims
        case device of
          Device {deviceType = CPU, deviceIndex = 0} -> do
            hfoldrM @IO DiagEmbedSpec () (hproduct (hproduct indexes dims) (hattach cpu (hproduct standardDTypes shapes)))
            hfoldrM @IO DiagEmbedSpec () (hproduct (hproduct indexes allDims) (hattach cpu (hproduct standardDTypes standardShapes)))
          Device {deviceType = CUDA, deviceIndex = 0} -> do
            hfoldrM @IO DiagEmbedSpec () (hproduct (hproduct indexes dims) (hattach cuda0 (hproduct (withHalf standardDTypes) shapes)))
            hfoldrM @IO DiagEmbedSpec () (hproduct (hproduct indexes allDims) (hattach cuda0 (hproduct (withHalf standardDTypes) standardShapes)))
      it "diagflat" $ do
        let shapes =
              standardShapes
                `happend` ( Proxy @'[0]
                              :. Proxy @'[1]
                              :. Proxy @'[2]
                              :. Proxy @'[0, 0]
                              :. Proxy @'[0, 1]
                              :. Proxy @'[1, 0]
                              :. HNil
                          )
            indexes = Proxy @0 :. Proxy @1 :. HNil
        case device of
          Device {deviceType = CPU, deviceIndex = 0} -> do
            hfoldrM @IO DiagflatSpec () (hproduct indexes (hattach cpu (hproduct standardDTypes shapes)))
          Device {deviceType = CUDA, deviceIndex = 0} -> do
            hfoldrM @IO DiagflatSpec () (hproduct indexes (hattach cuda0 (hproduct (withHalf standardDTypes) shapes)))
      it "diagonal" $ do
        let shapes1 = Proxy @'[2, 5, 4, 2] :. HNil
            shapes2 = Proxy @'[2, 3] :. shapes1
            allShapes = Proxy @'[1, 0] :. Proxy @'[0, 1] :. Proxy @'[1, 0] :. Proxy @'[2, 3] :. shapes2
            tris = Proxy @'Upper :. Proxy @'Lower :. HNil
            indexes = Proxy @0 :. HNil
            allIndexes = Proxy @1 :. indexes
            dims = (Proxy @0, Proxy @1) :. HNil
            allDims = (Proxy @0, Proxy @2) :. dims
        case device of
          Device {deviceType = CPU, deviceIndex = 0} -> do
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct indexes dims)) (hattach cpu (hproduct standardDTypes allShapes)))
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct allIndexes allDims)) (hattach cpu (hproduct standardDTypes shapes1)))
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct allIndexes dims)) (hattach cpu (hproduct standardDTypes shapes2)))
          Device {deviceType = CUDA, deviceIndex = 0} -> do
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct indexes dims)) (hattach cuda0 (hproduct (withHalf standardDTypes) allShapes)))
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct allIndexes allDims)) (hattach cuda0 (hproduct (withHalf standardDTypes) shapes1)))
            hfoldrM @IO DiagonalSpec () (hproduct (hproduct tris (hproduct allIndexes dims)) (hattach cuda0 (hproduct (withHalf standardDTypes) shapes2)))
