{-# LANGUAGE ConstraintKinds #-}
module NNSpec (spec) where

import Foreign hiding (void)
import Control.Monad (void)
import Foreign.C.Types

import Test.Hspec

import Torch.Types.THC

import qualified Torch.FFI.TH.Random as R
import qualified Torch.FFI.THC.General as General

import qualified Torch.FFI.THC.NN.Double as D
import qualified Torch.FFI.THC.Double.Tensor as D
import qualified Torch.FFI.THC.Double.TensorMath as D
import qualified Torch.FFI.THC.Double.TensorMathReduce as D
import qualified Torch.FFI.THC.Double.TensorRandom as D

{-
import Torch.FFI.THCUNN.Float as F
import Torch.FFI.THC.Float.Tensor as F
import Torch.FFI.THC.Float.TensorMath as F
import Torch.FFI.THC.Float.TensorRandom as F
-}

import Torch.FFI.TestsNN

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  s <- runIO (General.c_THCState_alloc >>= \s -> General.c_THCudaInit s >> pure s)
  afterAll_ (void $ General.c_THCState_free s) $ do
    -- describe "Float NNs"  $ testSuite nullPtr floatBook
    describe "Double NNs" $ testSuite s doubleBook


doubleBook :: NNTestSuite (Ptr C'THCState) (Ptr C'THCudaDoubleTensor) CDouble CDouble (Ptr ())
doubleBook = NNTestSuite
  { _newWithSize1d = D.c_newWithSize1d
  , _newWithSize2d = D.c_newWithSize2d
  , _newGen = pure nullPtr
  , _normal = Right D.c_normal
  , _fill = D.c_fill
  , _sumall = D.c_sumall
  , _free = D.c_free
  , _nnAbsUpdateOutput = D.c_Abs_updateOutput
  , _nnHSUpdateOutput = Nothing
  , _nnL1UpdateOutput = D.c_L1Cost_updateOutput
  , _nnRReLUUpdateOutput = D.c_RReLU_updateOutput
  }

{-
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
-}
