{-# LANGUAGE NoMonomorphismRestriction #-}

module ScriptSpec(spec) where

import Prelude hiding (abs, exp, floor, log, min, max)

import Test.Hspec
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.Layout
import Torch.TensorFactories
import Torch.Functional
import Torch.TensorOptions
import Torch.Script

spec :: Spec
spec = do
  it "define and run" $ do
    m <- newModule "m"
    define m $
      "def foo(self, x):\n" ++ 
      "    return (1, 2, x + 3)\n" ++ 
      "\n" ++ 
      "def forward(self, x):\n" ++ 
      "    tuple = self.foo(x)\n" ++ 
      "    return tuple\n"
    IVTuple [IVInt a,IVInt b,IVTensor c] <- run_method1 m "forward" (IVTensor (ones' []))
    a `shouldBe` 1
    b `shouldBe` 2
    (asValue c :: Float) `shouldBe` 4.0
