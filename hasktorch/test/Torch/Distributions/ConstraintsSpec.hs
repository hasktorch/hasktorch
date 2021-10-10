{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Torch.Distributions.ConstraintsSpec (spec) where

import GHC.Exts
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.Distributions.Constraints as Constraints
import qualified Torch.Functional as F
import qualified Torch.Tensor as D
import Torch.Typed.Tensor

spec :: Spec
spec = do
  it "boolean" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.boolean
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, True, False]

  it "integerInterval" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[4] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.integerInterval 1 2
            . D.asTensor
            $ [0, 1, 2, 3 :: Int]
    toList (Just t) `shouldBe` [False, True, True, False]

  it "integerGreaterThan" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[4] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.integerGreaterThan 1
            . D.asTensor
            $ [0, 1, 2, 3 :: Int]
    toList (Just t) `shouldBe` [False, False, True, True]

  it "real" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.real
            $ D.asTensor $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, True, True]
    let nans = F.divScalar (0.0 :: Float) $ D.asTensor [0.0, 1.0, 2.0 :: Float]
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.real
            $ nans
    toList (Just t) `shouldBe` [False, False, False]

  it "greaterThan" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.greaterThan 0.0
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [False, True, True]

  it "greaterThanEq" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.greaterThanEq 1.0
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [False, True, True]

  it "lessThan" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.lessThan 1.0
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, False, False]

  it "lessThanEq" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.lessThanEq 1.0
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, True, False]

  it "interval" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[4] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.interval 1.0 2.0
            . D.asTensor
            $ [0.0, 1.0, 2.0, 3.0 :: Float]
    toList (Just t) `shouldBe` [False, True, True, False]

  it "halfOpenInterval" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.halfOpenInterval 1.0 2.0
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [False, True, False]

  it "nonNegativeInteger" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.nonNegativeInteger
            . D.asTensor
            $ [-1, 0, 1 :: Int]
    toList (Just t) `shouldBe` [False, True, True]

  it "positiveInteger" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.positiveInteger
            . D.asTensor
            $ [0, 1, 2 :: Int]
    toList (Just t) `shouldBe` [False, True, True]

  it "integerInterval" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[4] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.integerInterval 1 2
            . D.asTensor
            $ [0, 1, 2, 3 :: Int]
    toList (Just t) `shouldBe` [False, True, True, False]

  it "positive" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[3] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.positive
            . D.asTensor
            $ [0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [False, True, True]

  it "unitInterval" $ do
    let t :: Tensor '( 'D.CPU, 0) 'D.Bool '[4] =
          toDevice @'( 'D.CPU, 0) . UnsafeMkTensor
            . Constraints.unitInterval
            . D.asTensor
            $ [-1.0, 0.0, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [False, True, True, False]
