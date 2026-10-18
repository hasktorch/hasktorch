{-# LANGUAGE ScopedTypeVariables #-}

module RandomSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.Device
import Torch.Random
import Torch.Tensor
import Torch.TensorOptions

spec :: Spec
spec = do
  it "pure functional random with seed" $ do
    generator <- mkGenerator (Device CPU 0) 0
    let (t, next) = randn' [4] generator
        (_, next') = randn' [4] next
        (t2, next'') = randn' [4] next'
        (t3, _) = randn' [5] generator
    shape t2 `shouldBe` [4]
    ((asValue t) :: [Float]) `shouldBe` take 4 (asValue t3)
  it "multinomial is a pure function of the generator value" $ do
    g <- mkPureGenerator (Device CPU 0) 0
    let probs = asTensor ([0.1, 0.2, 0.3, 0.4] :: [Float])
        (a, _) = multinomial probs 5 True g
        (b, _) = multinomial probs 5 True g
    (asValue a :: [Int]) `shouldBe` (asValue b :: [Int])
  it "multinomial is reproducible across generators from one seed" $ do
    g1 <- mkPureGenerator (Device CPU 0) 0
    g2 <- mkPureGenerator (Device CPU 0) 0
    let probs = asTensor ([0.1, 0.2, 0.3, 0.4] :: [Float])
        (a1, g1') = multinomial probs 5 True g1
        (b1, _) = multinomial probs 5 True g1'
        (a2, g2') = multinomial probs 5 True g2
        (b2, _) = multinomial probs 5 True g2'
    (asValue a1 :: [Int], asValue b1 :: [Int])
      `shouldBe` (asValue a2 :: [Int], asValue b2 :: [Int])
  it "multinomial respects the distribution" $ do
    g <- mkPureGenerator (Device CPU 0) 0
    let probs = asTensor ([0, 1, 0, 0] :: [Float])
        (t, _) = multinomial probs 10 True g
    (asValue t :: [Int]) `shouldBe` replicate 10 1
  it "restoring a PureGenerator state resumes the exact stream position" $ do
    g <- mkPureGenerator (Device CPU 0) 0
    let probs = asTensor ([0.1, 0.2, 0.3, 0.4] :: [Float])
        (_, g1) = multinomial probs 5 True g
    st <- getPureGeneratorState g1
    let (a, _) = multinomial probs 5 True g1
    g1' <- setPureGeneratorState g st
    let (b, _) = multinomial probs 5 True g1'
    (asValue a :: [Int]) `shouldBe` (asValue b :: [Int])
  it "restoring a Generator state resumes the exact stream position" $ do
    g <- mkGenerator (Device CPU 0) 0
    let (_, g1) = randn' [4] g
    st <- getGeneratorState g1
    let (a, _) = randn' [4] g1
    fresh <- mkGenerator (Device CPU 0) 1
    g1' <- setGeneratorState fresh st
    let (b, _) = randn' [4] g1'
    (asValue a :: [Float]) `shouldBe` (asValue b :: [Float])
