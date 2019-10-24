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

module Typed.TensorSpec
  ( spec
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
import           Torch.Typed.Factories
import           Torch.Typed.Native
import           Torch.Typed.Tensor

checkDynamicTensorAttributes
  :: forall device dtype shape
   . (TensorOptions shape dtype device)
  => Tensor device dtype shape
  -> IO ()
checkDynamicTensorAttributes t = do
  D.device untyped `shouldBe` optionsRuntimeDevice @shape @dtype @device
  D.dtype  untyped `shouldBe` optionsRuntimeDType  @shape @dtype @device
  D.shape  untyped `shouldBe` optionsRuntimeShape  @shape @dtype @device
 where untyped = toDynamic t

data UnarySpec = SinSpec

instance (TensorOptions shape dtype device)
  => Apply
       UnarySpec
       (Proxy (Tensor device dtype shape))
       (() -> IO ())
 where
  apply SinSpec _ _ = do
    let t = sin zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t

data BinarySpec = AddSpec | SubSpec | MulSpec

instance ( TensorOptions shape   dtype   device
         , TensorOptions shape'  dtype'  device'
         , TensorOptions shape'' dtype'' device'
         , device ~ device'
         , shape'' ~ Broadcast shape shape'
         , dtype'' ~ DTypePromotion dtype dtype'
         )
  => Apply
       BinarySpec
       (Proxy (Tensor device dtype shape), Proxy (Tensor device' dtype' shape'))
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
         )
  => Apply
       MatmulSpec
       (Proxy (Tensor device dtype shape), Proxy (Tensor device' dtype' shape'))
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
         )
  => Apply
       BinaryCmpSpec
       (Proxy (Tensor device dtype shape), Proxy (Tensor device' dtype' shape'))
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

allFloatingPointDTypes :: forall device shape . _
allFloatingPointDTypes =
  withHalf @device @shape (standardFloatingPointDTypes @device @shape)

standardFloatingPointDTypes :: forall device shape . _
standardFloatingPointDTypes =
  Proxy @(Tensor device 'D.Float shape)
    :. Proxy @(Tensor device 'D.Double shape)
    :. HNil

allDTypes :: forall device shape . _
allDTypes =
  withBool @device @shape (withHalf @device @shape (standardDTypes @device @shape))

withBool :: forall device shape . (forall device' shape' . _) -> _
withBool dtypes = Proxy @(Tensor device 'D.Bool shape) :. dtypes @device @shape

withHalf :: forall device shape . (forall device' shape' . _) -> _
withHalf dtypes = Proxy @(Tensor device 'D.Half shape) :. dtypes @device @shape

standardDTypes :: forall device shape . _
standardDTypes =
  Proxy @(Tensor device 'D.UInt8 shape)
    :. Proxy @(Tensor device 'D.Int8 shape)
    :. Proxy @(Tensor device 'D.Int16 shape)
    :. Proxy @(Tensor device 'D.Int32 shape)
    :. Proxy @(Tensor device 'D.Int64 shape)
    :. standardFloatingPointDTypes @device @shape

spec :: Spec
spec = do
  it "sin" (hfoldrM @IO SinSpec () (standardFloatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]))
  it "ones" $ do
    let t = ones :: CPUTensor 'D.Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros" $ do
    let t = zeros :: CPUTensor 'D.Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros with double" $ do
    let t = zeros :: CPUTensor 'D.Double '[2,3]
    checkDynamicTensorAttributes t
  it "randn" $ do
    t <- randn :: IO (CPUTensor 'D.Double '[2,3])
    checkDynamicTensorAttributes t
  let identicalShapes = hCartesianProduct (standardDTypes @'( 'D.CPU, 0) @'[2, 3]) (standardDTypes @'( 'D.CPU, 0) @'[2, 3])
      broadcastableShapes = hCartesianProduct (standardDTypes @'( 'D.CPU, 0) @'[3, 1, 4, 1]) (standardDTypes @'( 'D.CPU, 0) @'[2, 1, 1])
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
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[3]) (standardDTypes @'( 'D.CPU, 0) @'[3])
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[3, 2]) (standardDTypes @'( 'D.CPU, 0) @'[2, 4])
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[3]) (standardDTypes @'( 'D.CPU, 0) @'[3, 4])
        hfoldrM @IO MatmulSpec () shapes
      it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[3, 4]) (standardDTypes @'( 'D.CPU, 0) @'[4])
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[2, 1, 4, 3]) (standardDTypes @'( 'D.CPU, 0) @'[3, 3, 2])
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[3]) (standardDTypes @'( 'D.CPU, 0) @'[2, 3, 4])
        hfoldrM @IO MatmulSpec () shapes
      it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
        let shapes = hZipList (standardDTypes @'( 'D.CPU, 0) @'[2, 3, 4]) (standardDTypes @'( 'D.CPU, 0) @'[4])
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
  describe "eyeSquare" $ it "works" $ do
    let t = eyeSquare @2 :: CPUTensor 'D.Float '[2, 2]
    checkDynamicTensorAttributes t
    D.asValue (toDynamic t) `shouldBe` ([[1, 0], [0, 1]] :: [[Float]])
  describe "maxPool2d" $ it "works" $ do
    let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
    checkDynamicTensorAttributes c

