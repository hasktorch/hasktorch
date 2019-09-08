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

module StaticSpec
  ( spec
  )
where

import           Test.Hspec
import           Test.QuickCheck
import           Control.Exception.Safe

import           Prelude                 hiding ( sin )
import           Data.Reflection
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functions               as D
import qualified Torch.DType                   as D
import           Torch.Static
import           Torch.Static.Factories
import           Torch.Static.Native
import qualified Torch.TensorOptions           as D

spec :: Spec
spec = do
  it "sin" $ do
    let t = sin zeros :: Tensor Float '[2, 3]
    shape t `shouldBe` [2, 3]
    dtype t `shouldBe` D.Float
  it "ones" $ do
    let t = ones :: Tensor Float '[2, 3]
    shape t `shouldBe` [2, 3]
    dtype t `shouldBe` D.Float
  it "zeros" $ do
    let t = zeros :: Tensor Float '[2, 3]
    shape t `shouldBe` [2, 3]
    dtype t `shouldBe` D.Float
  it "zeros with double" $ do
    let t = zeros :: Tensor Double '[2, 3]
    shape t `shouldBe` [2, 3]
    dtype t `shouldBe` D.Double
  it "randn" $ do
    t <- randn :: IO (Tensor Double '[2, 3])
    shape t `shouldBe` [2, 3]
    dtype t `shouldBe` D.Double
  describe "add" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = add a b
      D.asValue (toDynamic c) `shouldBe` ([[2, 2, 2], [2, 2, 2]] :: [[Float]])
      shape c `shouldBe` [2, 3]
      dtype c `shouldBe` D.Float
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = add a b
      D.asValue (toDynamic c) `shouldBe` ([[2, 2], [2, 2], [2, 2]] :: [[Float]])
      shape c `shouldBe` [3, 2, 4, 1]
      dtype c `shouldBe` D.Float
  describe "sub" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = sub a b
      D.asValue (toDynamic c) `shouldBe` ([[0, 0, 0], [0, 0, 0]] :: [[Float]])
      shape c `shouldBe` [2, 3]
      dtype c `shouldBe` D.Float
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = sub a b
      D.asValue (toDynamic c) `shouldBe` ([[0, 0], [0, 0], [0, 0]] :: [[Float]])
      shape c `shouldBe` [3, 2, 4, 1]
      dtype c `shouldBe` D.Float
  describe "mul" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = mul a b
      D.asValue (toDynamic c) `shouldBe` ([[1, 1, 1], [1, 1, 1]] :: [[Float]])
      shape c `shouldBe` [2, 3]
      dtype c `shouldBe` D.Float
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = mul a b
      D.asValue (toDynamic c) `shouldBe` ([[1, 1], [1, 1], [1, 1]] :: [[Float]])
      shape c `shouldBe` [3, 2, 4, 1]
      dtype c `shouldBe` D.Float
  describe "matmul" $
    it "works" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[3, 1]
      let c = matmul a b
      D.asValue (toDynamic c) `shouldBe` ([[3],[3]] :: [[Float]])
      shape c `shouldBe` [2, 1]
      dtype c `shouldBe` D.Float
  describe "eyeSquare" $
    it "works" $ do
      let t = eyeSquare :: Tensor Float '[2, 2, 2]
      D.asValue (toDynamic t) `shouldBe` ([[[1]]] :: [[[Float]]])
      shape t `shouldBe` [2,2,2]
      dtype t `shouldBe` D.Float
