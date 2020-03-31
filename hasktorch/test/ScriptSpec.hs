{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}

module ScriptSpec(spec) where

import Prelude hiding (abs, exp, floor, log, min, max)

import Test.Hspec

import Torch
import Torch.Script
import Torch.NN
import Torch.Autograd
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


data MonoSpec = MonoSpec deriving (Show, Eq)

data MonoP = MonoP {
  m :: Parameter
  } deriving (Generic, Show)

instance Parameterized MonoP
instance Randomizable MonoSpec MonoP where
  sample MonoSpec  = do
    m <- makeIndependent (ones' [])
    return $ MonoP{..}

monop :: MonoP -> Tensor -> Tensor
monop MonoP{..} input = input * (toDependent m)

spec :: Spec
spec = describe "torchscript" $ do
  it "define and run" $ do
    let v00 = asTensor (4::Float)
    m <- newModule "m"
    v00' <- makeIndependent v00
    registerParameter m "p0" (toDependent v00') False
    define m $
      "def foo(self, x):\n" ++ 
      "    return (1, 2, x + 3 + self.p0)\n" ++ 
      "\n" ++ 
      "def forward(self, x):\n" ++ 
      "    tuple = self.foo(x)\n" ++ 
      "    return tuple\n"
    sm <- toScriptModule m
    let IVTuple [IVInt a,IVInt b,IVTensor c] = runMethod1 sm "forward" (IVTensor (ones' []))
    a `shouldBe` 1
    b `shouldBe` 2
    (asValue c :: Float) `shouldBe` 8.0
    sm2@(UnsafeScriptModule sm2') <- load "self.pt"
    p0 <- makeIndependent (asTensor (3::Float))
    setParameters (UnsafeRawModule sm2') [(toDependent p0)]
    let IVTuple [IVInt a,IVInt b,IVTensor c] = runMethod1 sm2 "forward" (IVTensor (ones' []))
    let [g] =  grad c (flattenParameters sm2)
    (asValue g :: Float) `shouldBe` 1.0
    return ()
    
  it "trace" $ do
    let v00 = asTensor (4::Float)
        v01 = asTensor (8::Float)
    m <- trace "MyModule" "forward" (\[x,y] -> return [x+y]) [v00,v01]
    sm <- toScriptModule m
    save sm "self2.pt"
    let (IVTensor r0) = forward sm (map IVTensor [v00,v01])
    (asValue r0::Float) `shouldBe` 12
  it "trace mlp with parameters" $ do
    v00 <- randnIO' [3,784]
    init' <- sample (MLPSpec 784 64 32 10)
    m <- traceWithParameters "MyModule" (\p [x] -> return [(mlp p x)]) init' [v00]
    sm <- toScriptModule m
    save sm "mlp.pt"
    let (IVTensor r0) = forward sm (map IVTensor [v00])
    (shape r0) `shouldBe` [3,10]
  it "trace monop with parameters" $ do
    let v00 = asTensor (4::Float)
    init' <- sample MonoSpec
    m <- traceWithParameters "MyModule" (\p [x] -> return [(monop p x)]) init' [v00]
    sm <- toScriptModule m
    save sm "monop.pt"
    let (IVTensor r0) = forward sm (map IVTensor [v00])
    (asValue r0::Float) `shouldBe` 4.0
    (shape r0) `shouldBe` []
    let p0 = asTensor (2::Float)
    rm <- toRawModule sm
    print $ (map asValue [p0] :: [Float])
    setParameters rm [p0]
    ps <- getParametersIO rm
    print $ (map asValue ps :: [Float])
    sm2 <- toScriptModule rm
    let (IVTensor r2) = forward sm2 (map IVTensor [v00])
    (asValue r2::Float) `shouldBe` 8.0
    (shape r2) `shouldBe` []
  it "run" $ do
    m2 <- load "self2.pt"
    let v10 = asTensor (40::Float)
        v11 = asTensor (80::Float)
    let (IVTensor r1) = forward m2 (map IVTensor [v10,v11])
    (asValue r1::Float) `shouldBe` 120
