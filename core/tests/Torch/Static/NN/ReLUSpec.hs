{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Static.NN.ReLUSpec where

import Data.Function ((&))
import Numeric.Backprop
import Test.Hspec

import Torch.Double
import Torch.Double.NN.Linear
import Torch.Double.NN.Activation

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "relu applied to a rank-1 tensor" $ do
    let t = cat1d
              (constant   5  :: Tensor '[2])
              (constant (-1) :: Tensor '[2])

    it "zeros out all < 0 values on the forward pass" $ do
      let (o, _) = backprop relu t
      tensordata o >>= (`shouldBe` [5, 5, 0, 0])

    it "returns exactly 0 for all < 0 on the backward pass" $ do
      let (_, g) = backprop relu t
      tensordata g >>= (`shouldBe` [1, 1, 0, 0])

  describe "linear to relu" $ do
    describe "with input that drops all values via ReLU" $ do
      let w = transpose2d $ unsafeMatrix [replicate 4 (-4), replicate 4 4]
          i = unsafeVector [1, 1, 1, 1]
          ll = Linear (w, constant 0) :: Linear 4 2
          (o',(lg', gin')) = backprop2 (relu .: linear 1) ll i
          (o ,(lg , gin )) = backprop2 (linear 1) ll i

      it "performs matrix multipication as you would expect" $ do
        tensordata o            >>= (`shouldBe` [-16,16])
      it "performs matrix multipication as you would expect" $ do
        tensordata (weights lg) >>= (`shouldBe` replicate 8 1)
      it "performs matrix multipication as you would expect" $ do
        tensordata (bias lg)    >>= (`shouldBe` [0,0])
      it "hello" $ do
        tensordata (addmv 1 (constant 0) 1 (transpose2d w) i) >>= (`shouldBe` [-16,16])

