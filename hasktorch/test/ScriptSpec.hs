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
spec = describe "torchscript" $ do
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
  it "trace" $ do
    let v00 = asTensor (4::Float)
        v01 = asTensor (8::Float)
    m <- trace (\[x,y] -> return [x+y]) [v00,v01]
    save m "self2.pt"
    (IVTensor r0) <- forward m (map IVTensor [v00,v01])
    (asValue r0::Float) `shouldBe` 12
  it "run" $ do
    m2 <- load "self2.pt"
    let v10 = asTensor (40::Float)
        v11 = asTensor (80::Float)
    (IVTensor r1) <- forward m2 (map IVTensor [v10,v11])
    (asValue r1::Float) `shouldBe` 120
