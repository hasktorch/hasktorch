{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Static.NN.ReLUSpec where

import Data.Function ((&))
import Numeric.Backprop
import Test.Hspec
import qualified Data.List as List

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
    describe "input that drops all values via ReLU" $ do
      let w = transpose2d $ unsafeMatrix [replicate 4 (-4), replicate 4 4]
          i = unsafeVector [1, 1, 1, 1]
          outputVal = [0,16]
      describe "with backprop'd Linear" $ do
        let ll = Linear (w, constant 0) :: Linear 4 2
            (o ,(lg , gin )) = backprop2 (relu .: linear 1) ll i

        it "performs matrix multipication as you would expect" $ do
          tensordata o >>= (`shouldBe` outputVal)

        it "returns the weight gradient as a matrix of 1s" $
          tensordata (weights lg) >>= (`shouldBe`
            concat (List.transpose
              [ replicate 4 0
              , replicate 4 1]))

        it "returns the bias gradient as a vector of 1s" $
          tensordata (bias lg)    >>= (`shouldBe` [0,1])

      describe "with Blas.addmv" $ do
        it "is the same as with backprop" $ do
          let t = addmv 1 (constant 0) 1 (transpose2d w) i
              t' = evalBP relu t
          tensordata t  >>= (`shouldBe` [-16,16])
          tensordata t' >>= (`shouldBe` outputVal)

