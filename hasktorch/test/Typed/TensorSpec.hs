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
  => Apply UnarySpec (Proxy (Tensor device dtype shape)) (() -> IO ())
 where
  apply SinSpec _ _ = do
    let t = sin zeros :: Tensor device dtype shape
    checkDynamicTensorAttributes t

data BinarySpec = AddSpec

instance ( TensorOptions shape   dtype   device
         , TensorOptions shape'  dtype'  device'
         , TensorOptions shape'' dtype'' device'
         , device ~ device'
         , shape'' ~ Broadcast shape shape'
         , dtype'' ~ DTypePromotion dtype dtype'
         )
  => Apply BinarySpec (Proxy (Tensor device dtype shape), Proxy (Tensor device' dtype' shape')) (() -> IO ())
 where
  apply AddSpec _ _ = do
    let a = ones @shape  @dtype  @device
    let b = ones @shape' @dtype' @device'
    let c = add a b
    checkDynamicTensorAttributes c

floatingPointDTypes :: forall device shape . _
floatingPointDTypes =
  Proxy @(Tensor device 'D.Half shape)
    :. Proxy @(Tensor device 'D.Float shape)
    :. Proxy @(Tensor device 'D.Double shape)
    :. HNil

allDTypes :: forall device shape . _
allDTypes =
  Proxy @(Tensor device 'D.Bool shape)
    :. Proxy @(Tensor device 'D.UInt8 shape)
    :. Proxy @(Tensor device 'D.Int8 shape)
    :. Proxy @(Tensor device 'D.Int16 shape)
    :. Proxy @(Tensor device 'D.Int32 shape)
    :. Proxy @(Tensor device 'D.Int64 shape)
    :. Proxy @(Tensor device 'D.Half shape)
    :. Proxy @(Tensor device 'D.Float shape)
    :. Proxy @(Tensor device 'D.Double shape)
    :. HNil

spec :: Spec
spec = do
  it "sin" (hfoldrM SinSpec () (floatingPointDTypes @'( 'D.CPU, 0) @'[2, 3]) :: IO ())
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
  describe "add" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = add a b :: CPUTensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2, 2, 2], [2, 2, 2]] :: [[Float]])
    it "works on tensors of identical shapes"
      (hfoldrM AddSpec () ((Proxy @(Tensor '( 'D.CPU, 0) 'D.Int64 '[2, 3]), Proxy @(Tensor '( 'D.CPU, 0) 'D.Int64 '[2, 3])) :. HNil) :: IO ())
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = add a b :: CPUTensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]]] :: [[[[Float]]]])
  describe "sub" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = sub a b :: CPUTensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[0, 0, 0], [0, 0, 0]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = sub a b :: CPUTensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]]] :: [[[[Float]]]])
  describe "mul" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = mul a b :: CPUTensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[1, 1, 1], [1, 1, 1]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = mul a b :: CPUTensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]]] :: [[[[Float]]]])
  describe "gt" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = gt a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = gt a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "lt" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = lt a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = lt a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "ge" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = ge a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = ge a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "le" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = le a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = le a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "eq" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = eq a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = eq a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "ne" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3]
      let b = ones :: CPUTensor 'D.Float '[2, 3]
      let c = ne a b :: CPUTensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: CPUTensor 'D.Float '[2, 1, 1]
      let c = ne a b :: CPUTensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "matmul" $ do
    it "returns the dot product if both tensors are 1-dimensional" $ do
      let a = ones :: CPUTensor 'D.Float '[3]
      let b = ones :: CPUTensor 'D.Float '[3]
      let c = matmul a b :: CPUTensor 'D.Float '[]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` (3 :: Float)
    it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 2]
      let b = ones :: CPUTensor 'D.Float '[2, 4]
      let c = matmul a b :: CPUTensor 'D.Float '[3, 4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2,2,2,2],[2,2,2,2],[2,2,2,2]] :: [[Float]])
    it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
      let a = ones :: CPUTensor 'D.Float '[3]
      let b = ones :: CPUTensor 'D.Float '[3, 4]
      let c = matmul a b :: CPUTensor 'D.Float '[4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([3,3,3,3] :: [Float])
    it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
      let a = ones :: CPUTensor 'D.Float '[3, 4]
      let b = ones :: CPUTensor 'D.Float '[4]
      let c = matmul a b :: CPUTensor 'D.Float '[3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([4,4,4] :: [Float])
    it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 1, 4, 3]
      let b = ones :: CPUTensor 'D.Float '[3, 3, 2]
      let c = matmul a b :: CPUTensor 'D.Float '[2, 3, 4, 2]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]]],[[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]]]] :: [[[[Float]]]])
    it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
      let a = ones :: CPUTensor 'D.Float '[3]
      let b = ones :: CPUTensor 'D.Float '[2, 3, 4]
      let c = matmul a b :: CPUTensor 'D.Float '[2, 4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[3,3,3,3],[3,3,3,3]] :: [[Float]])
    it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
      let a = ones :: CPUTensor 'D.Float '[2, 3, 4]
      let b = ones :: CPUTensor 'D.Float '[4]
      let c = matmul a b :: CPUTensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[4,4,4],[4,4,4]] :: [[Float]])
  describe "eyeSquare" $ it "works" $ do
    let t = eyeSquare @2 :: CPUTensor 'D.Float '[2, 2]
    checkDynamicTensorAttributes t
    D.asValue (toDynamic t) `shouldBe` ([[1, 0], [0, 1]] :: [[Float]])
  describe "maxPool2d" $ it "works" $ do
    let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones :: CPUTensor 'D.Float '[1,3,4,5])
    checkDynamicTensorAttributes c

