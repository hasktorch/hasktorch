{-# LANGUAGE ForeignFunctionInterface #-}

module TestMath (testMath) where

import Data.Maybe (fromJust)

import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import THFloatTensor
import THFloatTensorMath
import THFloatTensorRandom

import THIntTensor
import THIntTensorMath
import THIntTensorRandom

import THRandom
import Foreign.C.Types
import Test.Hspec

testMath = do
  hspec $ do
    describe "Math" $ do
      it "Can initialize values with the fill method" $ do
        t1 <- c_THDoubleTensor_newWithSize2d 2 2
        c_THDoubleTensor_fill t1 3.1
        c_THDoubleTensor_get2d t1 0 0 `shouldBe` (3.1 :: CDouble)
        c_THDoubleTensor_free t1
      it "Can invert double values with cinv" $ do
        t1 <- c_THDoubleTensor_newWithSize2d 3 2
        c_THDoubleTensor_fill t1 2.0
        result <- c_THDoubleTensor_newWithSize2d 3 2
        c_THDoubleTensor_cinv result t1
        c_THDoubleTensor_get2d result 0 0 `shouldBe` (0.5 :: CDouble)
        c_THDoubleTensor_get2d t1 0 0 `shouldBe` (2.0 :: CDouble)
        c_THDoubleTensor_free t1
        c_THDoubleTensor_free result

      -- cinv doesn't seem to be excluded by the preprocessor, yet is not implemented
      -- for Int
      -- it "Can invert int values with cinv (?)" $ do
      --   t1 <- c_THIntTensor_newWithSize2d 3 2
      --   c_THIntTensor_fill t1 2
      --   result <- c_THIntTensor_newWithSize2d 3 2
      --   c_THIntTensor_cinv result t1
      --   c_THIntTensor_get2d result 0 0 `shouldBe` (0 :: CInt)
      --   c_THIntTensor_get2d t1 0 0 `shouldBe` (2 :: CInt)
      --   c_THIntTensor_free t1
      --   c_THIntTensor_free result




