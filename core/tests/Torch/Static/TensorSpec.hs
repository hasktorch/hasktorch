module Torch.Static.TensorSpec where

import Test.Hspec
import Torch.Double

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "un/squeezing" $ do
    let d1 = dim :: Dim 1
    it "unsqueeze adds a dimension" $ do
      shape (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5])) `shouldBe` [3,1,4,5]

    it "squeeze removes a dimension" $ do
      shape (squeeze1d d1 (constant 50 :: Tensor '[3,1,4,5])) `shouldBe` [3,4,5]

    it "unsqueeze+squeeze forms an identity" $ do
      shape (squeeze1d d1 (unsqueeze1d d1 (constant 50 :: Tensor '[3,4,5]))) `shouldBe` [3,4,5]

  describe "stacking" $ do
    it "works with two identical tensors" $ do
      let t = constant (-(1/4)) :: Tensor '[6]
      tensordata (stack1d0 t t) >>= (`shouldBe` replicate 12 (-1/4))

    it "works with two differing tensors" $ do
      let t1 = constant (-(1/4)) :: Tensor '[6]
          t2 = constant   (1/4)  :: Tensor '[6]
      tensordata (stack1d0 t1 t2) >>= (`shouldBe` (replicate 6 (-1/4) ++ replicate 6 (1/4)))




