{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.FFI.TestsNN where

import Foreign
import Foreign.C.Types

import Test.Hspec

type NNNum r = (Fractional r, Show r, Eq r)

data NNTestSuite state ten real accreal gen = NNTestSuite
  { _newWithSize1d :: state -> CLLong -> IO ten
  , _newWithSize2d :: state -> CLLong -> CLLong -> IO ten
  , _newGen :: IO gen
  , _normal :: Either (state -> ten -> gen -> CDouble -> CDouble -> IO ()) (state -> ten -> CDouble -> CDouble -> IO ())
  , _fill   :: state -> ten -> real -> IO ()
  , _sumall :: state -> ten -> IO accreal
  , _free   :: state -> ten -> IO ()
  , _nnAbsUpdateOutput :: state -> ten -> ten -> IO ()
  , _nnHSUpdateOutput :: Maybe (state -> ten -> ten -> CDouble -> IO ())
  , _nnL1UpdateOutput :: state -> ten -> ten -> IO ()
  , _nnRReLUUpdateOutput :: state -> ten -> ten -> ten -> CDouble -> CDouble -> CBool -> CBool -> gen -> IO ()
  }

testSuite :: (NNNum real, NNNum accreal) => state -> NNTestSuite state ten real accreal gen -> Spec
testSuite s fs = do
  it "Abs test" $ do
    t1 <- newWithSize2d s 2 2
    fill s t1 (-3)
    nnAbsUpdateOutput s t1 t1
    sumall s t1 >>= (`shouldBe` 12.0)
    free s t1
  case mnnHSUpdateOutput of
    Nothing -> pure ()
    Just nnHSUpdateOutput -> 
      it "HardShrink test" $ do
        t1 <- newWithSize2d s 2 2
        t2 <- newWithSize2d s 2 2
        fill s t2 4
        fill s t1 4
        nnHSUpdateOutput s t1 t1 100.0
        sumall s t1 >>= (`shouldBe` 0.0)
        nnHSUpdateOutput s t2 t2 1.0
        sumall s t2 >>= (`shouldBe` 16.0)
        free s t1
        free s t2
  it "L1Cost_updateOutput" $ do
    t1 <- newWithSize1d s 1
    fill s t1 3
    nnL1UpdateOutput s t1 t1
    sumall s t1 >>= (`shouldBe` 3.0)
    free s t1
  it "RReLU_updateOutput" $ do
    t1 <- newWithSize1d s 100
    t2 <- newWithSize1d s 100
    fill s t2 0.5
    g <- newGen
    case enormal of
      (Left  normal) -> normal s t1 g 0 1
      (Right normal) -> normal s t1   0 1
    nnRReLUUpdateOutput s t2 t2 t1 0.0 15.0 1 1 g
    sumall s t2 >>= (`shouldBe` 50.0)
    free s t1
    free s t2

 where
  newWithSize1d = _newWithSize1d fs
  newWithSize2d = _newWithSize2d fs
  newGen = _newGen fs
  enormal = _normal fs
  fill = _fill fs
  sumall = _sumall fs
  free = _free fs
  nnAbsUpdateOutput = _nnAbsUpdateOutput fs
  mnnHSUpdateOutput = _nnHSUpdateOutput fs
  nnL1UpdateOutput = _nnL1UpdateOutput fs
  nnRReLUUpdateOutput = _nnRReLUUpdateOutput fs


