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

module StaticSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Control.Exception.Safe

import Prelude hiding (sin)
import qualified Torch.Tensor as D
import qualified Torch.TensorFactories as D
import qualified Torch.Functions as D
import qualified Torch.DType as D
import Torch.Static
import Torch.Static.Factories
import Torch.Static.Native
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
    let t = sin zeros :: Tensor Float '[2,3]
    checkDynamicTensorAttributes t
  it "ones" $ do
    let t = ones :: Tensor Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros" $ do
    let t = zeros :: Tensor Float '[2,3]
    checkDynamicTensorAttributes t
  it "zeros with double" $ do
    let t = zeros :: Tensor Double '[2,3]
    checkDynamicTensorAttributes t
  it "randn" $ do
    t <- randn :: IO (Tensor Double '[2,3])
    checkDynamicTensorAttributes t
  describe "add" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = add a b :: Tensor Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2, 2, 2], [2, 2, 2]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = add a b :: Tensor Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[2, 2], [2, 2], [2, 2]] :: [[Float]])
      D.asValue (toDynamic c) `shouldBe` ([[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]],[[[2],[2],[2],[2]],[[2],[2],[2],[2]]]] :: [[[[Float]]]])
  describe "sub" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = sub a b :: Tensor Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[0, 0, 0], [0, 0, 0]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = sub a b :: Tensor Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[0, 0], [0, 0], [0, 0]] :: [[Float]])
      D.asValue (toDynamic c) `shouldBe` ([[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]],[[[0],[0],[0],[0]],[[0],[0],[0],[0]]]] :: [[[[Float]]]])
  describe "mul" $ do
    it "works on tensors of identical shapes" $ do
      let a = ones :: Tensor Float '[2, 3]
      let b = ones :: Tensor Float '[2, 3]
      let c = mul a b :: Tensor Float '[2, 3]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[1, 1, 1], [1, 1, 1]] :: [[Float]])
    it "works on broadcastable tensors of different shapes" $ do
      let a = ones :: Tensor Float '[3, 1, 4, 1]
      let b = ones :: Tensor Float '[2, 1, 1]
      let c = mul a b :: Tensor Float '[3, 2, 4, 1]
      checkDynamicTensorAttributes c
      D.asValue (toDynamic c) `shouldBe` ([[1, 1], [1, 1], [1, 1]] :: [[Float]])
      D.asValue (toDynamic c) `shouldBe` ([[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]],[[[1],[1],[1],[1]],[[1],[1],[1],[1]]]] :: [[[[Float]]]])
