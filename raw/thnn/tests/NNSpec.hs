{-# LANGUAGE ConstraintKinds #-}
module NNSpec (spec) where

import Foreign
import Foreign.C.Types

import Test.Hspec

import Torch.Types.TH

import Torch.FFI.THNN.Double as D
import Torch.FFI.TH.Double.Tensor as D
import Torch.FFI.TH.Double.TensorMath as D
import Torch.FFI.TH.Double.TensorRandom as D

import Torch.FFI.THNN.Float as F
import Torch.FFI.TH.Float.Tensor as F
import Torch.FFI.TH.Float.TensorMath as F
import Torch.FFI.TH.Float.TensorRandom as F

import Torch.FFI.TH.Random as R

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Float NNs"  floatSpec
  describe "Double NNs" doubleSpec

data NNTestSuite state ten real accreal = NNTestSuite
  { _newWithSize1d :: state -> CLLong -> IO ten
  , _newWithSize2d :: state -> CLLong -> CLLong -> IO ten
  , _newGen :: IO (Ptr C'THGenerator)
  , _normal :: ten -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()
  , _fill   :: state -> ten -> real -> IO ()
  , _sumall :: state -> ten -> IO accreal
  , _free   :: state -> ten -> IO ()
  , _nnAbsUpdateOutput :: state -> ten -> ten -> IO ()
  , _nnHSUpdateOutput :: state -> ten -> ten -> CDouble -> IO ()
  , _nnL1UpdateOutput :: state -> ten -> ten -> IO ()
  , _nnRReLUUpdateOutput :: state -> ten -> ten -> ten -> CDouble -> CDouble -> CBool -> CBool -> Ptr C'THGenerator -> IO ()
  }

type NNNum r = (Fractional r, Show r, Eq r)

testSuite :: (NNNum real, NNNum accreal) => state -> NNTestSuite state ten real accreal -> Spec
testSuite s fs = do
  it "Abs test" $ do
    t1 <- newWithSize2d s 2 2
    fill s t1 (-3)
    nnAbsUpdateOutput s t1 t1
    sumall s t1 >>= (`shouldBe` 12.0)
    free s t1
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
    gen <- newGen
    normal t1 gen 0.0 1.0
    nnRReLUUpdateOutput s t2 t2 t1 0.0 15.0 1 1 gen
    sumall s t2 >>= (`shouldBe` 50.0)
    free s t1
    free s t2
 where
  newWithSize1d = _newWithSize1d fs
  newWithSize2d = _newWithSize2d fs
  newGen = _newGen fs
  normal = _normal fs
  fill = _fill fs
  sumall = _sumall fs
  free = _free fs
  nnAbsUpdateOutput = _nnAbsUpdateOutput fs
  nnHSUpdateOutput = _nnHSUpdateOutput fs
  nnL1UpdateOutput = _nnL1UpdateOutput fs
  nnRReLUUpdateOutput = _nnRReLUUpdateOutput fs


doubleSpec :: Spec
doubleSpec = testSuite nullPtr doubleBook

doubleBook :: NNTestSuite (Ptr C'THNNState) (Ptr C'THDoubleTensor) CDouble CDouble
doubleBook = NNTestSuite
  { _newWithSize1d = D.c_newWithSize1d
  , _newWithSize2d = D.c_newWithSize2d
  , _newGen = R.c_THGenerator_new
  , _normal = D.c_normal
  , _fill = D.c_fill
  , _sumall = D.c_sumall
  , _free = D.c_free
  , _nnAbsUpdateOutput = D.c_Abs_updateOutput
  , _nnHSUpdateOutput = D.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = D.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = D.c_RReLU_updateOutput
  }

floatSpec :: Spec
floatSpec = testSuite nullPtr floatBook

floatBook :: NNTestSuite (Ptr C'THNNState) (Ptr C'THFloatTensor) CFloat CDouble
floatBook = NNTestSuite
  { _newWithSize1d = F.c_newWithSize1d
  , _newWithSize2d = F.c_newWithSize2d
  , _newGen = R.c_THGenerator_new
  , _normal = F.c_normal
  , _fill = F.c_fill
  , _sumall = F.c_sumall
  , _free = F.c_free
  , _nnAbsUpdateOutput = F.c_Abs_updateOutput
  , _nnHSUpdateOutput = F.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = F.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = F.c_RReLU_updateOutput
  }

