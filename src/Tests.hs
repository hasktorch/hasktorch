{-# LANGUAGE ForeignFunctionInterface #-}

module Main where

import Torch as TR

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.ForeignPtr
import Foreign.Marshal.Array

import Test.Hspec

tests = do
  hspec $ do
    describe "Torch" $ do
      it "initializes empty tensor with 0 dimension" $ do
        t0 <- c_THFloatTensor_new
        TR.c_THFloatTensor_nDimension t0 `shouldBe` 0
      it "initializes tensor with 1 dimension" $ do
        t1 <- c_THFloatTensor_newWithSize1d 10
        TR.c_THFloatTensor_nDimension t1 `shouldBe` 1
      it "initializes 1D tensor with correct size" $ do
        t1 <- TR.c_THFloatTensor_newWithSize1d 10
        (TR.c_THFloatTensor_size t1 0) `shouldBe` 10

main = do
  tests
  putStrLn "Done"
