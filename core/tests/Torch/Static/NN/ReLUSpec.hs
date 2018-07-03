{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Static.NN.ReLUSpec where

import Test.Hspec
import Numeric.Backprop
import Torch.Double
import Torch.Double.NN.Linear
import Torch.Double.NN.Activation

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "relu applied to a rank-1 tensor" $ do
    t <-
      runIO $ cat1d
        (constant   5  :: Tensor '[2])
        (constant (-1) :: Tensor '[2])

    it "zeros out all < 0 values on the forward pass" $ do
      let (o, _) = backprop relu t
      tensordata o >>= (`shouldBe` [5, 5, 0, 0])

    it "returns exactly 0 for all < 0 on the backward pass" $ do
      let (_, g) = backprop relu t
      tensordata g >>= (`shouldBe` [1, 1, 0, 0])

