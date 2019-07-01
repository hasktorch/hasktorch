{-# LANGUAGE NoMonomorphismRestriction #-}

module SparseSpec(spec) where

import Prelude hiding (abs, exp, floor, log, min, max)

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions

spec :: Spec
spec = do
  it "create sparse tensor" $ do
    let i = [[0, 1, 1],
             [2, 0, 2]] :: [[Integer]]
        v = [3, 4, 5] :: [Double]
    let x = sparseCooTensor' (asConstTensor i) (asConstTensor v) [2, 3]
    print (toDense x)
    -- When we call print for sparse tensor, it throws a exception.
    (print x) `shouldThrow` anyException
    (asValue (toDense x) :: [[Double]]) `shouldBe` [[0.0,0.0,3.0],[4.0,0.0,5.0]]
    (asValue (toDense (x+x)) :: [[Double]]) `shouldBe` [[0.0,0.0,6.0],[8.0,0.0,10.0]]
    (asValue (toDense (toSparse (toDense (x+x)))) :: [[Double]]) `shouldBe` [[0.0,0.0,6.0],[8.0,0.0,10.0]]
