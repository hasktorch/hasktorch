module FactorySpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.DType
import Torch.Functional
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions

spec :: Spec
spec = do
  it "ones factory" $ do
    let x = ones' [50]
    shape x `shouldBe` [50]
  it "zeros factory" $ do
    let x = zeros' [50]
    shape x `shouldBe` [50]
  it "onesLike factory" $ do
    let x = onesLike $ zeros' [50]
    shape x `shouldBe` [50]
  it "zerosLike factory" $ do
    let x = zerosLike $ ones' [50]
    shape x `shouldBe` [50]
  it "randIO factory" $ do
    x <- randIO' [50]
    shape x `shouldBe` [50]
  it "randnIO factory" $ do
    x <- randnIO' [50]
    shape x `shouldBe` [50]
  it "linspace factory" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = linspace start end 50 defaultOpts
    (toDouble $ select 0 49 x) `shouldBe` 25.0
  it "logspace factory" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = logspace start end 50 2.0 defaultOpts
    (toDouble $ select 0 0 x) `shouldBe` 32.0
  it "eyeSquare factory" $ do
    let x = eyeSquare' 7
    shape x `shouldBe` [7, 7]
    (toDouble $ select 0 0 (select 0 0 x)) `shouldBe` 1.0
    (toDouble $ select 0 1 (select 0 0 x)) `shouldBe` 0.0
  it "eye factory" $ do
    let x = eye' 7 3
    shape x `shouldBe` [7, 3]
    (toDouble $ select 0 0 (select 0 0 x)) `shouldBe` 1.0
    (toDouble $ select 0 1 (select 0 0 x)) `shouldBe` 0.0
  it "full factory" $ do
    let x = full' [5, 2] (15.0 :: Double)
    shape x `shouldBe` [5, 2]
    (toDouble $ select 0 0 (select 0 0 x)) `shouldBe` 15.0
  it "arange factory" $ do
    let x = arange' 0 10 2
    shape x `shouldBe` [5]
    asValue x `shouldBe` [0 :: Float, 2, 4, 6, 8]
