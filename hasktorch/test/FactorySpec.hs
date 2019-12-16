module FactorySpec (spec) where

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functional
import Torch.TensorOptions

spec :: Spec
spec = do
  it "ones factory" $ do
    let x = ones' [50]
    shape x `shouldBe` [50]
  it "zeros factory" $ do
    let x = zeros' [50]
    shape x `shouldBe` [50]
  it "rand factory" $ do
    x <- rand' [50]
    shape x `shouldBe` [50]
  it "randn factory" $ do
    x <- randn' [50]
    shape x `shouldBe` [50]
  it "linspace factory" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = linspace  start end 50 defaultOpts
    (toDouble $ select x 0 49) `shouldBe` 25.0
  it "logspace factory" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = logspace start end 50 2.0 defaultOpts
    (toDouble $ select x 0 0) `shouldBe` 32.0
  it "eyeSquare factory" $ do
    let x = eyeSquare' 7
    shape x `shouldBe` [7, 7]
    (toDouble $ select (select x 0 0) 0 0) `shouldBe` 1.0
    (toDouble $ select (select x 0 0) 0 1) `shouldBe` 0.0
  it "eye factory" $ do
    let x = eye' 7 3
    shape x `shouldBe` [7, 3]
    (toDouble $ select (select x 0 0) 0 0) `shouldBe` 1.0
    (toDouble $ select (select x 0 0) 0 1) `shouldBe` 0.0
  it "full factory" $ do
    let x = full' [5, 2] (15.0 :: Double)
    shape x `shouldBe` [5, 2]
    (toDouble $ select (select x 0 0) 0 0) `shouldBe` 15.0
