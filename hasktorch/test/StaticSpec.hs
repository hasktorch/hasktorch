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

spec :: Spec
spec = do
  it "sin" $ do
    let t = sin zeros :: Tensor Float '[2,3]
    D.shape (toDynamic t) `shouldBe` [2,3]
    D.dtype (toDynamic t) `shouldBe` D.Float
  it "ones" $ do
    let t = ones :: Tensor Float '[2,3]
    D.shape (toDynamic t) `shouldBe` [2,3]
    D.dtype (toDynamic t) `shouldBe` D.Float
  it "zeros" $ do
    let t = zeros :: Tensor Float '[2,3]
    D.shape (toDynamic t) `shouldBe` [2,3]
    D.dtype (toDynamic t) `shouldBe` D.Float
  it "zeros with double" $ do
    let t = zeros :: Tensor Double '[2,3]
    D.shape (toDynamic t) `shouldBe` [2,3]
    D.dtype (toDynamic t) `shouldBe` D.Double
  it "randn" $ do
    t <- randn :: IO (Tensor Double '[2,3])
    D.shape (toDynamic t) `shouldBe` [2,3]
    D.dtype (toDynamic t) `shouldBe` D.Double
  it "add" $ do
    let a = ones :: Tensor Float '[2,3]
    let b = ones :: Tensor Float '[2,3]
    let c = a + b :: Tensor Float '[2,3]
    D.asValue (toDynamic c) `shouldBe` ([[2,2,2],[2,2,2]]::[[Float]])
    D.shape (toDynamic c) `shouldBe` [2,3]
    D.dtype (toDynamic c) `shouldBe` D.Float
