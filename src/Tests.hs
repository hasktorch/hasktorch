{-# LANGUAGE ForeignFunctionInterface #-}

module Main where

import THFloatTensor as TR
import Foreign.C.Types

import Test.Hspec

tests :: IO ()
tests = do
  hspec $ do
    describe "Tensor creation and access methods" $ do
      it "initializes empty tensor with 0 dimension" $ do
        t <- c_THFloatTensor_new
        TR.c_THFloatTensor_nDimension t `shouldBe` 0
        c_THFloatTensor_free t
      it "1D tensor has correct dimensions and sizes" $ do
        t <- TR.c_THFloatTensor_newWithSize1d 10
        TR.c_THFloatTensor_nDimension t `shouldBe` 1
        TR.c_THFloatTensor_size t 0 `shouldBe` 10
        c_THFloatTensor_free t
      it "2D tensor has correct dimensions and sizes" $ do
        t <- c_THFloatTensor_newWithSize2d 10 25
        TR.c_THFloatTensor_nDimension t `shouldBe` 2
        TR.c_THFloatTensor_size t 0 `shouldBe` 10
        TR.c_THFloatTensor_size t 1 `shouldBe` 25
        c_THFloatTensor_free t
      it "3D tensor has correct dimensions and sizes" $ do
        t <- c_THFloatTensor_newWithSize3d 10 25 5
        TR.c_THFloatTensor_nDimension t `shouldBe` 3
        TR.c_THFloatTensor_size t 0 `shouldBe` 10
        TR.c_THFloatTensor_size t 1 `shouldBe` 25
        TR.c_THFloatTensor_size t 2 `shouldBe` 5
        c_THFloatTensor_free t
      it "4D tensor has correct dimensions and sizes" $ do
        t <- c_THFloatTensor_newWithSize4d 10 25 5 62
        TR.c_THFloatTensor_nDimension t `shouldBe` 4
        TR.c_THFloatTensor_size t 0 `shouldBe` 10
        TR.c_THFloatTensor_size t 1 `shouldBe` 25
        TR.c_THFloatTensor_size t 2 `shouldBe` 5
        TR.c_THFloatTensor_size t 3 `shouldBe` 62
        c_THFloatTensor_free t
      it "Can assign and retrieve correct 1D vector values" $ do
        t <- TR.c_THFloatTensor_newWithSize1d 10
        c_THFloatTensor_set1d t 0 (CFloat 20.5)
        c_THFloatTensor_set1d t 1 (CFloat 1.4)
        c_THFloatTensor_set1d t 9 (CFloat 3.14)
        c_THFloatTensor_get1d t 0 `shouldBe` (20.5 :: Float)
        c_THFloatTensor_get1d t 1 `shouldBe` (1.4 :: Float)
        c_THFloatTensor_get1d t 9 `shouldBe` (3.14 :: Float)
        c_THFloatTensor_free t
      it "Can assign and retrieve correct 2D vector values" $ do
        t <- TR.c_THFloatTensor_newWithSize2d 10 15
        c_THFloatTensor_set2d t 0 0 (CFloat 20.5)
        c_THFloatTensor_set2d t 1 5 (CFloat 1.4)
        c_THFloatTensor_set2d t 9 9 (CFloat 3.14)
        c_THFloatTensor_get2d t 0 0 `shouldBe` (20.5 :: Float)
        c_THFloatTensor_get2d t 1 5 `shouldBe` (1.4 :: Float)
        c_THFloatTensor_get2d t 9 9 `shouldBe` (3.14 :: Float)
        c_THFloatTensor_free t
      it "Can assign and retrieve correct 3D vector values" $ do
        t <- TR.c_THFloatTensor_newWithSize3d 10 15 10
        c_THFloatTensor_set3d t 0 0 0 (CFloat 20.5)
        c_THFloatTensor_set3d t 1 5 3 (CFloat 1.4)
        c_THFloatTensor_set3d t 9 9 9 (CFloat 3.14)
        c_THFloatTensor_get3d t 0 0 0 `shouldBe` (20.5 :: Float)
        c_THFloatTensor_get3d t 1 5 3 `shouldBe` (1.4 :: Float)
        c_THFloatTensor_get3d t 9 9 9 `shouldBe` (3.14 :: Float)
        c_THFloatTensor_free t
      it "Can assign and retrieve correct 4D vector values" $ do
        t <- TR.c_THFloatTensor_newWithSize4d 10 15 10 20
        c_THFloatTensor_set4d t 0 0 0 0 (CFloat 20.5)
        c_THFloatTensor_set4d t 1 5 3 2 (CFloat 1.4)
        c_THFloatTensor_set4d t 9 9 9 9 (CFloat 3.14)
        c_THFloatTensor_get4d t 0 0 0 0 `shouldBe` (20.5 :: Float)
        c_THFloatTensor_get4d t 1 5 3 2 `shouldBe` (1.4 :: Float)
        c_THFloatTensor_get4d t 9 9 9 9 `shouldBe` (3.14 :: Float)
        c_THFloatTensor_free t



main :: IO ()
main = do
  tests
  putStrLn "Done"
