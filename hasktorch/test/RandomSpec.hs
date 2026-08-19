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
