{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}

module ScriptSpec(spec) where

import Prelude hiding (abs, exp, floor, log, min, max)

import Test.Hspec

import Torch
import Torch.Script
import GHC.Generics

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP {
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = 
    logSoftmax 1
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

params :: Parameterized a => a -> [Tensor]
params = (map toDependent).flattenParameters

fromParams :: Parameterized a => a -> [Tensor] -> a
fromParams init' ps = replaceParameters init' (map IndependentTensor ps)

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
    sm <- toScriptModule m
    let IVTuple [IVInt a,IVInt b,IVTensor c] = runMethod1 sm "forward" (IVTensor (ones' []))
    a `shouldBe` 1
    b `shouldBe` 2
    (asValue c :: Float) `shouldBe` 4.0
  it "trace" $ do
    let v00 = asTensor (4::Float)
        v01 = asTensor (8::Float)
    m <- trace "MyModule" "forward" (\[x,y] -> return [x+y]) [v00,v01]
    sm <- toScriptModule m
    save sm "self2.pt"
    let (IVTensor r0) = forward sm (map IVTensor [v00,v01])
    (asValue r0::Float) `shouldBe` 12
  it "trace mlp" $ do
    v00 <- randnIO' [3,784]
    init' <- sample (MLPSpec 784 64 32 10)
    m <- trace "MyModule" "forward" (\(x:p) -> return [(mlp (fromParams init' p) x)]) (v00:params init')
    sm <- toScriptModule m
    save sm "mlp.pt"
    let (IVTensor r0) = forward sm (map IVTensor (v00:params init'))
    (shape r0) `shouldBe` [3,10]
  it "trace mlp with parameters" $ do
    v00 <- randnIO' [3,784]
    init' <- sample (MLPSpec 784 64 32 10)
    m <- traceWithParameters "MyModule" (\p [x] -> return [(mlp (fromParams init' p) x)]) (params init') [v00]
    sm <- toScriptModule m
    save sm "mlp2.pt"
    let (IVTensor r0) = forward sm (map IVTensor [v00])
    (shape r0) `shouldBe` [3,10]
  it "run" $ do
    m2 <- load "self2.pt"
    let v10 = asTensor (40::Float)
        v11 = asTensor (80::Float)
    let (IVTensor r1) = forward m2 (map IVTensor [v10,v11])
    (asValue r1::Float) `shouldBe` 120
