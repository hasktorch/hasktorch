{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}

module Torch.Typed.TensorSpec
  ( Torch.Typed.TensorSpec.spec
  )
where

import           Prelude                 hiding ( sin )
import           Control.Exception.Safe
import           Foreign.Storable
import           Data.HList
import           Data.Proxy
import           Data.Reflection

import           Test.Hspec
import           Test.QuickCheck

import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.Functions               as D
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import           Torch.Typed.Aux
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor
import           Torch.Typed.AuxSpec

data BinarySpec = AddSpec | SubSpec | MulSpec

instance ( TensorOptions shape   dtype   device
         , TensorOptions shape'  dtype'  device'
         , TensorOptions shape'' dtype'' device'
         , device ~ device'
         , shape'' ~ Broadcast shape shape'
         , dtype'' ~ DTypePromotion dtype dtype'
         , DTypeIsNotBool dtype,   DTypeIsNotHalf dtype
         , DTypeIsNotBool dtype',  DTypeIsNotHalf dtype'
         , DTypeIsNotBool dtype'', DTypeIsNotHalf dtype''
         )
  => Apply
       BinarySpec
       ((Proxy device, (Proxy dtype, Proxy shape)), (Proxy device', (Proxy dtype', Proxy shape')))
       (() -> IO ())
 where
  apply AddSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = add a b
    checkDynamicTensorAttributes c
  apply SubSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = sub a b
    checkDynamicTensorAttributes c
  apply MulSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = mul a b
    checkDynamicTensorAttributes c

data MatmulSpec = MatmulSpec

instance ( TensorOptions shape   dtype  device
         , TensorOptions shape'  dtype' device'
         , TensorOptions shape'' dtype' device'
         , device ~ device'
         , dtype ~ dtype'
         , shape'' ~ MatMul shape shape'
         , DTypeIsNotBool dtype, DTypeIsNotHalf dtype
         )
  => Apply
       MatmulSpec
       ((Proxy device, (Proxy dtype, Proxy shape)), (Proxy device', (Proxy dtype', Proxy shape')))
       (() -> IO ())
 where
  apply MatmulSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = matmul a b
    checkDynamicTensorAttributes c

data BinaryCmpSpec = GTSpec | LTSpec | GESpec | LESpec | EQSpec | NESpec

instance ( TensorOptions shape   dtype  device
         , TensorOptions shape'  dtype' device'
         , TensorOptions shape'' D.Bool device'
         , device ~ device'
         , shape'' ~ Broadcast shape shape'
         , DTypeIsNotBool dtype,  DTypeIsNotHalf dtype
         , DTypeIsNotBool dtype', DTypeIsNotHalf dtype'
         )
  => Apply
       BinaryCmpSpec
       ((Proxy device, (Proxy dtype, Proxy shape)), (Proxy device', (Proxy dtype', Proxy shape')))
       (() -> IO ())
 where
  apply GTSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = gt a b
    checkDynamicTensorAttributes c
  apply LTSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = lt a b
    checkDynamicTensorAttributes c
  apply GESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = ge a b
    checkDynamicTensorAttributes c
  apply LESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = le a b
    checkDynamicTensorAttributes c
  apply EQSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = eq a b
    checkDynamicTensorAttributes c
  apply NESpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = ne a b
    checkDynamicTensorAttributes c

spec :: Spec
spec = do
  let standardShapes = Proxy @'[2, 3] :. HNil
      broadcastableShapes0 = Proxy @'[3, 1, 4, 1] :. HNil
      broadcastableShapes1 = Proxy @'[2, 1, 1] :. HNil
      standardDTypes' = hCartesianProduct3 justCPU standardDTypes standardShapes
      identicalShapes = hCartesianProduct standardDTypes' standardDTypes'
      broadcastableShapes = hCartesianProduct (hCartesianProduct3 justCPU standardDTypes broadcastableShapes0) (hCartesianProduct3 justCPU standardDTypes broadcastableShapes1)
  describe "binary operators" $ do
    describe "add" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO AddSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO AddSpec () broadcastableShapes)
    describe "sub" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO SubSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO SubSpec () broadcastableShapes)
    describe "mul" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO MulSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO MulSpec () broadcastableShapes)
    describe "matmul" $ do
      it "returns the dot product if both tensors are 1-dimensional" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3, 2] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[2, 4] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3, 4] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3, 4] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[4] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[2, 1, 4, 3] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3, 3, 2] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[3] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[2, 3, 4] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
        let shapes = hZipList (hCartesianProduct3 justCPU standardDTypes (Proxy @'[2, 3, 4] :. HNil)) (hCartesianProduct3 justCPU standardDTypes (Proxy @'[4] :. HNil))
        hfoldrM @IO MatmulSpec () shapes
  describe "binary comparison operators" $ do
    describe "gt" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO GTSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO GTSpec () broadcastableShapes)
    describe "lt" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO GTSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO GTSpec () broadcastableShapes)
    describe "ge" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO GESpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO GESpec () broadcastableShapes)
    describe "le" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO LESpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO LESpec () broadcastableShapes)
    describe "eq" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO EQSpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO EQSpec () broadcastableShapes)
    describe "ne" $ do
      it "works on tensors of identical shapes"
        (hfoldrM @IO NESpec () identicalShapes)
      it "works on broadcastable tensors of different shapes"
        (hfoldrM @IO NESpec () broadcastableShapes)
