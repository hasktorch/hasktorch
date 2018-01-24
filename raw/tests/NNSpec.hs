module NNSpec (spec) where

import Foreign.C.Types
import Foreign.Ptr

import Test.Hspec

import THFloatNN as T
import THDoubleNN as T
import THFloatTensor as T
import THFloatTensorMath as T
import THDoubleTensor as T
import THDoubleTensorMath as T


main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "testsFloat" testsFloat
  describe "testsDouble" testsDouble

testsFloat :: Spec
testsFloat =
  describe "Float NN" $ do
    it "initializes tensor and perform NN_abs operation" $ do
      t1 <- c_THFloatTensor_newWithSize2d 2 2
      T.c_THFloatTensor_fill t1 (3.0)
      T.c_THNN_FloatAbs_updateOutput nullPtr t1 t1
      T.c_THFloatTensor_sumall t1 `shouldBe` 12.0
      T.c_THFloatTensor_free t1

testsDouble :: Spec
testsDouble =
  describe "Double NN" $ do
    it "initializes tensor and perform NN_abs operation" $ do
      t1 <- c_THDoubleTensor_newWithSize2d 2 2
      T.c_THDoubleTensor_fill t1 (3.0)
      T.c_THNN_DoubleAbs_updateOutput nullPtr t1 t1
      T.c_THDoubleTensor_sumall t1 `shouldBe` 12.0
      T.c_THDoubleTensor_free t1
