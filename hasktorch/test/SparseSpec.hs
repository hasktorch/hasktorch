{-# LANGUAGE NoMonomorphismRestriction #-}

module SparseSpec (spec) where

import Control.Exception.Safe
import Test.Hspec
import Torch.DType
import Torch.Functional
import Torch.Layout
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions
import Prelude hiding (abs, exp, floor, log, max, min)

spec :: Spec
spec = do
  it "create sparse tensor" $ do
    let i =
          [ [0, 1, 1],
            [2, 0, 2]
          ] ::
            [[Int]]
        v = [3, 4, 5] :: [Float]
    let x = sparseCooTensor' (asTensor i) (asTensor v) [2, 3]
    (shape (asTensor i)) `shouldBe` [2, 3]
    (shape (asTensor v)) `shouldBe` [3]
    print (toDense x)
    -- When we call print for sparse tensor, it throws a exception.
    print x -- `shouldThrow` anyException
    (asValue (toDense x) :: [[Float]]) `shouldBe` [[0.0, 0.0, 3.0], [4.0, 0.0, 5.0]]
    (asValue (toDense (x + x)) :: [[Float]]) `shouldBe` [[0.0, 0.0, 6.0], [8.0, 0.0, 10.0]]
    (asValue (toDense (toSparse (toDense (x + x)))) :: [[Float]]) `shouldBe` [[0.0, 0.0, 6.0], [8.0, 0.0, 10.0]]
  it "zeros sparse tensor" $ do
    let x = zeros [2, 3] $ withLayout Sparse defaultOpts
    print x -- `shouldThrow` anyException
    (asValue (toDense x) :: [[Float]]) `shouldBe` [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  it "large sparse tensor" $ do
    let x = zeros [1000, 1000, 1000] $ withLayout Sparse defaultOpts
    shape x `shouldBe` [1000, 1000, 1000]
