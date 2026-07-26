{-# LANGUAGE RankNTypes #-}

-- | The staged element-wise layer, exercised from the untyped API: the same
-- polymorphic formula runs per element at @a = Float@ and vectorized at
-- @a = Tensor@, and the two agree.
module ElementwiseSpec (spec) where

import Test.Hspec
import Torch
import Torch.Elementwise

relu' :: (Floating a, Cond a) => a -> a
relu' x = whereE (gtE x 0) x 0

clamp01 :: (Floating a, Cond a) => a -> a
clamp01 x = minE 1 (maxE 0 x)

spec :: Spec
spec = do
  describe "Torch.Elementwise" $ do
    it "emap agrees with the Float instantiation of the same formula" $ do
      let t = asTensor ([-2, -0.5, 0, 0.5, 2] :: [Float])
      (asValue (emap relu' t) :: [Float]) `shouldBe` map relu' [-2, -0.5, 0, 0.5, 2]
      (asValue (emap clamp01 t) :: [Float]) `shouldBe` map clamp01 [-2, -0.5, 0, 0.5, 2]
    it "ezipWith computes an elementwise maximum via native maxE" $ do
      let x = asTensor ([1, 5, 3] :: [Float])
          y = asTensor ([4, 2, 3] :: [Float])
      (asValue (ezipWith maxE x y) :: [Float]) `shouldBe` [4, 5, 3]
    it "whereE keeps gradients flowing (autograd sees every step)" $ do
      t <- makeIndependent (asTensor ([-1, 2] :: [Float]))
      let loss = sumAll (emap relu' (toDependent t))
          [g] = grad loss [t]
      (asValue g :: [Float]) `shouldBe` [0, 1]
