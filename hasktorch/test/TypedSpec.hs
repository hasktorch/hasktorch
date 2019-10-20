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

module TypedSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Control.Exception.Safe

import Prelude hiding (sin)
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functions as D
import qualified Torch.DType as D
import Torch.Typed
import Torch.Typed.Factories
import Torch.Typed.Native
import qualified Torch.TensorOptions as D
import Data.Reflection

checkDynamicTensorAttributes
  :: forall dtype shape
    . (TensorOptions dtype shape)
  => Tensor dtype shape
  -> IO ()
checkDynamicTensorAttributes t = do
  shape t `shouldBe` optionsRuntimeShape @dtype @shape
  dtype t `shouldBe` optionsRuntimeDType @dtype @shape

spec :: Spec
spec = do
  it "sin" $ do
    let t = sin zeros :: Tensor 'D.Float '[2,3]
    checkDynamicTensorAttributes t
  it "ones" $ do
    let t = ones :: Tensor 'D.Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros" $ do
    let t = zeros :: Tensor 'D.Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros with double" $ do
    let t = zeros :: Tensor 'D.Double '[2,3]
    checkDynamicTensorAttributes t
  it "randn" $ do
    t <- randn :: IO (Tensor 'D.Double '[2,3])
    checkDynamicTensorAttributes t
  describe "add" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = add a b :: Tensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2, 2, 2], [2, 2, 2]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = add a b :: Tensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]]] :: [[[[Float]]]])
  describe "sub" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = sub a b :: Tensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[0, 0, 0], [0, 0, 0]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = sub a b :: Tensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]]] :: [[[[Float]]]])
  describe "mul" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = mul a b :: Tensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[1, 1, 1], [1, 1, 1]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = mul a b :: Tensor 'D.Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]]] :: [[[[Float]]]])
  describe "gt" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = gt a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = gt a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "lt" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = lt a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = lt a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "ge" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = ge a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = ge a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "le" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = le a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = le a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "eq" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = eq a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[True, True, True], [True, True, True]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = eq a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]],[[[True],[True],[True],[True]],[[True],[True],[True],[True]]]] :: [[[[Bool]]]])
  describe "ne" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor 'D.Float '[2, 3]
      let b = ones :: Tensor 'D.Float '[2, 3]
      let c = ne a b :: Tensor 'D.Bool '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[False, False, False], [False, False, False]] :: [[Bool]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor 'D.Float '[3, 1, 4, 1]
      let b = ones :: Tensor 'D.Float '[2, 1, 1]
      let c = ne a b :: Tensor 'D.Bool '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]],[[[False],[False],[False],[False]],[[False],[False],[False],[False]]]] :: [[[[Bool]]]])
  describe "matmul" $ do
    it "returns the dot product if both tensors are 1-dimensional" $ do
      let a = ones :: Tensor 'D.Float '[3]
      let b = ones :: Tensor 'D.Float '[3]
      let c = matmul a b :: Tensor 'D.Float '[]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` (3 :: Float)
    it "returns the matrix-matrix product if both arguments are 2-dimensional" $ do
      let a = ones :: Tensor 'D.Float '[3, 2]
      let b = ones :: Tensor 'D.Float '[2, 4]
      let c = matmul a b :: Tensor 'D.Float '[3, 4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2,2,2,2],[2,2,2,2],[2,2,2,2]] :: [[Float]])
    it "returns the matrix-matrix product if the first argument is 1-dimensional and the second argument is 2-dimensional by temporarily adding a 1 to the dimension of the first argument" $ do
      let a = ones :: Tensor 'D.Float '[3]
      let b = ones :: Tensor 'D.Float '[3, 4]
      let c = matmul a b :: Tensor 'D.Float '[4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([3,3,3,3] :: [Float])
    it "returns the matrix-vector product if the first argument is 2-dimensional and the second argument is 1-dimensional" $ do
      let a = ones :: Tensor 'D.Float '[3, 4]
      let b = ones :: Tensor 'D.Float '[4]
      let c = matmul a b :: Tensor 'D.Float '[3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([4,4,4] :: [Float])
    it "returns a batched matrix-matrix product if both arguments are at least 2-dimensional and the batch (i.e. non-matrix) dimensions are broadcastable" $ do
      let a = ones :: Tensor 'D.Float '[2, 1, 4, 3]
      let b = ones :: Tensor 'D.Float '[3, 3, 2]
      let c = matmul a b :: Tensor 'D.Float '[2, 3, 4, 2]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]]],[[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]],[[3,3],[3,3],[3,3],[3,3]]]] :: [[[[Float]]]])
    it "returns a batched matrix-matrix product if the first argument is 1-dimensional and the second argument has more than 2 dimensions" $ do
      let a = ones :: Tensor 'D.Float '[3]
      let b = ones :: Tensor 'D.Float '[2, 3, 4]
      let c = matmul a b :: Tensor 'D.Float '[2, 4]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[3,3,3,3],[3,3,3,3]] :: [[Float]])
    it "returns a batched matrix-vector product if the first argument has more than 2 dimensions and the second argument is 1-dimensional" $ do
      let a = ones :: Tensor 'D.Float '[2, 3, 4]
      let b = ones :: Tensor 'D.Float '[4]
      let c = matmul a b :: Tensor 'D.Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[4,4,4],[4,4,4]] :: [[Float]])
  describe "eyeSquare" $ it "works" $ do
    let t = eyeSquare @'D.Float @2
    checkDynamicTensorAttributes t
    D.asValue (toDynamic t) `shouldBe` ([[1, 0], [0, 1]] :: [[Float]])
  describe "maxPool2d" $ it "works" $ do
    let c = maxPool2d @'(1,1) @'(1,1) @'(0,0) (ones::Tensor 'D.Float '[1,3,4,5])
    checkDynamicTensorAttributes c

