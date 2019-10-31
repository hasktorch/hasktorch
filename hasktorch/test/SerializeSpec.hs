
module SerializeSpec(spec) where

import Test.Hspec
import System.Directory (removeFile)

import Torch.Tensor
import Torch.Serialize

spec :: Spec
spec = do
  it "save and load tensor" $ do
    let i = [[0, 1, 1],
             [2, 0, 2]] :: [[Int]]
        v = [3, 4, 5] :: [Float]
    save [(asTensor i),(asTensor v)] "test.pt"
    tensors <- load "test.pt"
    removeFile "test.pt"
    length tensors `shouldBe` 2
    let [ii,vv] = tensors
    (asValue ii :: [[Int]]) `shouldBe` i
    (asValue vv :: [Float]) `shouldBe` v
