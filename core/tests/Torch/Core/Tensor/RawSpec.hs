module Torch.Core.Tensor.RawSpec (spec) where

import Test.Hspec
import Foreign (Ptr)

import TensorTypes (TensorDim(..), TensorDoubleRaw)
import TensorRaw (tensorRaw)
import Torch.Core.Tensor.Raw (toList)


main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "toList" toListSpec


-- it would be nice to convert this into property checks
toListSpec :: Spec
toListSpec = do
  describe "0 dimensional tensors" $ do
    t <- runIO (mkTensor25 D0)
    it "returns the correct length"     $ length (toList t) `shouldBe` 0
    it "returns the correct values"     $ toList t   `shouldSatisfy` all (== 25)

  describe "1 dimensional tensors" $ do
    assertTmap (D1 5)

  describe "2 dimensional tensors" $ do
    assertTmap (D2 (2,5))

  describe "3 dimensional tensors" $
    assertTmap (D3 (4,2,5))

  describe "4 dimensional tensors" $
    assertTmap (D4 (8,4,2,5))

 where
  mkTensor25 :: TensorDim Word -> IO TensorDoubleRaw
  mkTensor25 = flip tensorRaw 25

  assertTmap :: TensorDim Word -> Spec
  assertTmap d = do
    t <- runIO (mkTensor25 d)
    it "returns the correct length"     $ length (toList t) `shouldBe` fromIntegral (product d)
    it "returns the correct values"     $ toList t   `shouldSatisfy` all (== 25)


