{-# LANGUAGE ConstraintKinds #-}
module NNSpec (spec) where

import Foreign
import Foreign.C.Types

import Test.Hspec

import Torch.Types.TH

import Torch.FFI.THCUNN.Double as D
import Torch.FFI.THC.Double.Tensor as D
import Torch.FFI.THC.Double.TensorMath as D
import Torch.FFI.THC.Double.TensorRandom as D

import Torch.FFI.THCUNN.Float as F
import Torch.FFI.THCU.Float.Tensor as F
import Torch.FFI.THCU.Float.TensorMath as F
import Torch.FFI.THCU.Float.TensorRandom as F

import Torch.FFI.THC.Random as R

import Torch.FFI.TestsNN 

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Float NNs"  $ testSuite nullPtr floatBook
  describe "Double NNs" $ testSuite nullPtr doubleBook

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

