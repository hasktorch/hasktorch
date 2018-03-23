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

import Torch.FFI.TestsNN

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Float NNs"  $ testSuite nullPtr floatBook
  describe "Double NNs" $ testSuite nullPtr doubleBook

doubleBook :: NNTestSuite (Ptr C'THNNState) (Ptr C'THDoubleTensor) CDouble CDouble (Ptr C'THGenerator)
doubleBook = NNTestSuite
  { _newWithSize1d = D.c_newWithSize1d
  , _newWithSize2d = D.c_newWithSize2d
  , _newGen = R.c_THGenerator_new
  , _normal = Just D.c_normal
  , _fill = D.c_fill
  , _sumall = D.c_sumall
  , _free = D.c_free
  , _nnAbsUpdateOutput = D.c_Abs_updateOutput
  , _nnHSUpdateOutput = Just D.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = D.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = D.c_RReLU_updateOutput
  }


floatBook :: NNTestSuite (Ptr C'THNNState) (Ptr C'THFloatTensor) CFloat CDouble (Ptr C'THGenerator)
floatBook = NNTestSuite
  { _newWithSize1d = F.c_newWithSize1d
  , _newWithSize2d = F.c_newWithSize2d
  , _newGen = R.c_THGenerator_new
  , _normal = Just F.c_normal
  , _fill = F.c_fill
  , _sumall = F.c_sumall
  , _free = F.c_free
  , _nnAbsUpdateOutput = F.c_Abs_updateOutput
  , _nnHSUpdateOutput = Just F.c_HardShrink_updateOutput
  , _nnL1UpdateOutput = F.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = F.c_RReLU_updateOutput
  }

