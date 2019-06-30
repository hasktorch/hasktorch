{-# LANGUAGE NoMonomorphismRestriction #-}

module FunctionsSpec(spec) where

import Prelude hiding (abs, floor, min, max)

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions

spec :: Spec
spec = do
  it "scales and adds" $ do
    let x = 2 * ones' [10]
    let y = 3 * ones' [10]
    let z = x + y
    (toDouble $ select z 0 4) `shouldBe` 5.0
  it "sumAll" $ do
    let x = 2 * ones' [5]
    let y = sumAll x
    toDouble y `shouldBe` 10.0
  it "abs" $ do
    let x = (-2) * ones' [5]
    let y = abs x
    (toDouble $ select y 0 0) `shouldBe` 2.0
  it "add" $ do
    let x = (-2) * ones' [5]
    let y = abs x
    let z = add x y
    (toDouble $ select z 0 0) `shouldBe` 0.0
  it "sub" $ do
    let x = (-2) * ones' [5]
    let y = abs x
    let z = sub x y
    (toDouble $ select z 0 0) `shouldBe` -4.0
  it "ceil" $ do
    x <- rand' [5]
    let y = ceil x
    (toDouble $ select y 0 0) `shouldBe` 1.0
  it "floor" $ do
    x <- rand' [5]
    let y = floor x
    (toDouble $ select y 0 0) `shouldBe` 0.0
  it "takes the minimum of a linspace" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = linspace start end 50 defaultOpts
    let m = min x
    toDouble m `shouldBe` 5.0
  it "takes the maximum of a linspace" $ do
    let start = 5.0 :: Double
    let end = 25.0 :: Double
    let x = linspace start end 50 defaultOpts
    let m = max x
    toDouble m `shouldBe` 25.0
  it "takes the median of a linspace" $ do
    let x = linspace (5.0 :: Double) (10.0 :: Double) 5 defaultOpts
    let m = median x
    toDouble m `shouldBe` 7.5
