module NNSpec (spec) where

import Foreign.C.Types
import Foreign.Ptr

import Test.Hspec

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

data NNTestSuite = NNTestSuite
  { _newWithSize1d :: Maybe ()
  , _newWithSize2d :: Maybe ()
  , _newGen :: Maybe ()
  , _normal :: Maybe ()
  , _fill :: Maybe ()
  , _sumall :: Maybe ()
  , _free :: Maybe ()
  , _nnABSUpdateOutput :: Maybe ()
  , _nnHSUpdateOutput :: Maybe ()
  , _nnL1UpdateOutput :: Maybe ()
  , _nnRReLUUpdateOutput :: Maybe ()
  }

testSuite :: NNTestSuite -> Spec
testSuite fs = do
  it "Abs test" $ do
    t1 <- newWithSize2d 2 2
    fill t1 (-3.0)
    nnABSUpdateOutput nullPtr t1 t1
    sumall t1 `shouldBe` 12.0
    free t1
  it "HardShrink test" $ do
    t1 <- newWithSize2d 2 2
    t2 <- newWithSize2d 2 2
    fill t2 4.0
    fill t1 4.0
    nnHSUpdateOutput nullPtr t1 t1 100.0
    sumall t1 `shouldBe` 0.0
    nnHSUpdateOutput nullPtr t2 t2 1.0
    sumall t2 `shouldBe` 16.0
    free t1
    free t2
  it "L1Cost_updateOutput" $ do
    t1 <- newWithSize1d 1
    fill t1 3
    nnL1UpdateOutput nullPtr t1 t1
    sumall t1 `shouldBe` 3.0
    free t1
  it "RReLU_updateOutput" $ do
    t1 <- newWithSize1d 100
    t2 <- newWithSize1d 100
    fill t2 0.5
    gen <- newGen
    normal t1 gen 0.0 1.0
    nnRReLUUpdateOutput nullPtr t2 t2 t1 0.0 15.0 1 1 gen
    sumall t2 `shouldBe` 50.0
    free t1
    free t2
 where
  newWithSize1d = _newWithSize1d fs
  newWithSize2d = _newWithSize2d fs
  newGen = _newGen fs
  normal = _normal fs
  fill = _fill fs
  sumall = _sumall fs
  free = _free fs
  nnABSUpdateOutput = _nnABSUpdateOutput fs
  nnHSUpdateOutput = _nnHSUpdateOutput fs
  nnL1UpdateOutput = _nnL1UpdateOutput fs
  nnRReLUUpdateOutput = _nnRReLUUpdateOutput fs


doubleSpec :: Spec
doubleSpec = testSuite NNTestSuite
  { _newWithSize1d = D.c_newWithSize1d
  , _newWithSize2d = D.c_newWithSize2d
  , _newGen = R.c_new
  , _normal = D.c_normal
  , _fill = D.c_fill
  , _sumall = D.c_sumall
  , _free = D.c_free
  , _nnABSUpdateOutput = D.c_AbsCriterion_updateOutput
  , _nnHSUpdateOutput = D.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = D.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = D.c_RReLU_updateOutput
  }

floatSpec :: Spec
floatSpec = testSuite NNTestSuite
  { _newWithSize1d = F.c_newWithSize1d
  , _newWithSize2d = F.c_newWithSize2d
  , _newGen = R.c_new
  , _normal = F.c_normal
  , _fill = F.c_fill
  , _sumall = F.c_sumall
  , _free = F.c_free
  , _nnABSUpdateOutput = F.c_AbsCriterion_updateOutput
  , _nnHSUpdateOutput = F.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = F.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = F.c_RReLU_updateOutput
  }

