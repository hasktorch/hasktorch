{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Distributions.CategoricalSpec (spec) where

import Test.Hspec
import GHC.Exts
import qualified Torch.Device as D
import qualified Torch.Tensor as D
import qualified Torch.DType as D
import qualified Torch.Functional as F
import qualified Torch.Distributions.Constraints as Constraints
import Torch.Typed.Tensor
import Torch.Distributions.Distribution
import Torch.Distributions.Categorical

type Tnsr dtype shape = Tensor '( 'D.CPU, 0) dtype shape

spec :: Spec
spec = do

  let ps = [0.8, 0.2 :: Float]
  let p = D.asTensor ps
  let d = fromProbs p

  it "batch_shape" $ do
    batch_shape d `shouldBe` []

  it "event_shape" $ do
    event_shape d `shouldBe` []

  it "probs" $ do
    -- putStrLn . show $ probs d
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ probs d
    toList (Just t) `shouldBe` ps

  it "expand" $ do
    -- putStrLn . show $ expand d [2]
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ probs $ expand d [2]
    toList (Just t) !! 0 `shouldBe` 0.8  -- ps

  it "support" $ do
    -- putStrLn . show $ support d $ D.asTensor [0.0, 0.5, 1.0, 2.0 :: Float]
    let t :: Tnsr 'D.Bool '[4] = UnsafeMkTensor .
            support d $ D.asTensor [0.0, 0.5, 1.0, 2.0 :: Float]
    toList (Just t) `shouldBe` [True, True, True, False]

  it "mean" $ do
    -- putStrLn . show $ mean d
    let t :: Tnsr 'D.Float '[] = UnsafeMkTensor $ mean d
    isInfinite (toFloat t) `shouldBe` True

  it "variance" $ do
    -- putStrLn . show $ variance d
    let t :: Tnsr 'D.Float '[] = UnsafeMkTensor $ variance d
    isInfinite (toFloat t) `shouldBe` True

  it "sample" $ do
    t <- sample d [2]
    -- putStrLn . show $ t
    D.shape t `shouldBe` [2]
    let t' :: Tnsr 'D.Bool '[2] = UnsafeMkTensor . Constraints.boolean $ t
    toList (Just t') `shouldBe` [True, True]

  it "sample: multi-dimensional" $ do
    let d = fromProbs . D.asTensor $ [ [0.3, 0.2], [0.4, 0.1 :: Float]]
    t <- sample d [3]
    -- putStrLn . show $ t
    D.shape t `shouldBe` [3,2]

  it "log_prob" $ do
    -- putStrLn . show $ log_prob d $ D.asTensor [0.3, 0.5 :: Float]
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ log_prob d $ D.asTensor [0.3, 0.5 :: Float]
    toList (Just t) `shouldBe` [-9.691001e-2, -9.691001e-2]

  it "entropy" $ do
    -- putStrLn . show $ entropy d
    let t :: Tnsr 'D.Float '[] = UnsafeMkTensor $ entropy d
    abs (toFloat t - 0.2173) < 0.01 `shouldBe` True

  it "enumerate_support" $ do
    -- putStrLn . show $ enumerate_support d False
    let t :: Tnsr 'D.Float '[2] = UnsafeMkTensor $ enumerate_support d False
    toList (Just t) `shouldBe` [0.0, 1.0]
