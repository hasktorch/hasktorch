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
import THFloatTensorRandom as T
import THDoubleTensorRandom as T
import THRandom as T


main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "testsFloat" testsFloat
  describe "testsDouble" testsDouble

testsFloat :: Spec
testsFloat =
  describe "Float NN" $ do
    it "Abs test" $ do
      t1 <- c_THFloatTensor_newWithSize2d 2 2
      T.c_THFloatTensor_fill t1 (-3.0)
      T.c_THNN_FloatAbs_updateOutput nullPtr t1 t1
      T.c_THFloatTensor_sumall t1 `shouldBe` 12.0
      T.c_THFloatTensor_free t1
    it "HardShrink test" $ do
      t1 <- c_THFloatTensor_newWithSize2d 2 2
      t2 <- c_THFloatTensor_newWithSize2d 2 2
      T.c_THFloatTensor_fill t2 4.0
      T.c_THFloatTensor_fill t1 4.0
      T.c_THNN_FloatHardShrink_updateOutput nullPtr t1 t1 100.0
      T.c_THFloatTensor_sumall t1 `shouldBe` 0.0
      T.c_THNN_FloatHardShrink_updateOutput nullPtr t2 t2 1.0
      T.c_THFloatTensor_sumall t2 `shouldBe` 16.0
      T.c_THFloatTensor_free t1
      T.c_THFloatTensor_free t2
    it "L1Cost_updateOutput" $ do
      t1 <- c_THFloatTensor_newWithSize1d 1
      T.c_THFloatTensor_fill t1 (3.0)
      T.c_THNN_FloatL1Cost_updateOutput nullPtr t1 t1
      T.c_THFloatTensor_sumall t1 `shouldBe` 3.0
      T.c_THFloatTensor_free t1
    it "RReLU_updateOutput" $ do
      t1 <- c_THFloatTensor_newWithSize1d 100
      t2 <- c_THFloatTensor_newWithSize1d 100
      T.c_THFloatTensor_fill t2 0.5
      gen <- T.c_THGenerator_new
      T.c_THFloatTensor_normal t1 gen 0.0 1.0
      T.c_THNN_FloatRReLU_updateOutput nullPtr t2 t2 t1 0.0 15.0 1 1 gen
      T.c_THFloatTensor_sumall t2 `shouldBe` 50.0
      T.c_THFloatTensor_free t1
      T.c_THFloatTensor_free t2

testsDouble :: Spec
testsDouble =
  describe "Double NN" $ do
    it "Abs test" $ do
      t1 <- c_THDoubleTensor_newWithSize2d 2 2
      T.c_THDoubleTensor_fill t1 (-3.0)
      T.c_THNN_DoubleAbs_updateOutput nullPtr t1 t1
      T.c_THDoubleTensor_sumall t1 `shouldBe` 12.0
      T.c_THDoubleTensor_free t1
    it "HardShrink test" $ do
      t1 <- c_THDoubleTensor_newWithSize2d 2 2
      t2 <- c_THDoubleTensor_newWithSize2d 2 2
      T.c_THDoubleTensor_fill t2 (4.0)
      T.c_THDoubleTensor_fill t1 (4.0)
      T.c_THNN_DoubleHardShrink_updateOutput nullPtr t1 t1 100.0
      T.c_THDoubleTensor_sumall t1 `shouldBe` 0.0
      T.c_THNN_DoubleHardShrink_updateOutput nullPtr t2 t2 1.0
      T.c_THDoubleTensor_sumall t2 `shouldBe` 16.0
      T.c_THDoubleTensor_free t1
      T.c_THDoubleTensor_free t2
    it "L1Cost_updateOutput" $ do
      t1 <- c_THDoubleTensor_newWithSize1d 1
      T.c_THDoubleTensor_fill t1 (3.0)
      T.c_THNN_DoubleL1Cost_updateOutput nullPtr t1 t1
      T.c_THDoubleTensor_sumall t1 `shouldBe` 3.0
      T.c_THDoubleTensor_free t1
    it "RReLU_updateOutput" $ do
      t1 <- c_THDoubleTensor_newWithSize1d 100
      t2 <- c_THDoubleTensor_newWithSize1d 100
      T.c_THDoubleTensor_fill t2 0.5
      gen <- T.c_THGenerator_new
      T.c_THDoubleTensor_normal t1 gen 0.0 1.0
      T.c_THNN_DoubleRReLU_updateOutput nullPtr t2 t2 t1 0.0 15.0 1 1 gen
      T.c_THDoubleTensor_sumall t2 `shouldBe` 50.0
      T.c_THDoubleTensor_free t1
      T.c_THDoubleTensor_free t2
