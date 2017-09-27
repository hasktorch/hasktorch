{-# LANGUAGE ForeignFunctionInterface #-}

module TestRandom (testsRandom) where

import Data.Maybe (fromJust)

import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import THFloatTensor
import THFloatTensorMath
import THFloatTensorRandom

import THRandom
import Foreign.C.Types
import Test.Hspec

import TorchTensor

-- testsFloat = do
--   gen <- c_THGenerator_new
--   hspec $ do
--     describe "random vectors" $ do
--       it "uniform random is < bound" $ do
--         t <- fromJust $ tensorFloatnew [1000]
--         c_THFloatTensor_uniform t gen (-1.0) (1.0)
--         (c_THFloatTensor_maxall t) `shouldSatisfy` (< 1.001)
--         c_THFloatTensor_free t
--       it "uniform random is > bound" $ do
--         t <- fromJust $ tensorFloatnew [1000]
--         c_THFloatTensor_uniform t gen (-1.0) (1.0)
--         (c_THFloatTensor_maxall t) `shouldSatisfy` (> (-1.001))
--         c_THFloatTensor_free t
--       it "normal mean follows law of large numbers" $ do
--         t <- fromJust $ tensorFloatnew [100000]
--         c_THFloatTensor_normal t gen 1.55 0.25
--         (c_THFloatTensor_meanall t) `shouldSatisfy` (\x -> and [(x < 1.6), (x > 1.5)])
--         c_THFloatTensor_free t
--       it "normal std follows law of large numbers" $ do
--         t <- fromJust $ tensorFloatnew [100000]
--         c_THFloatTensor_normal t gen 1.55 0.25
--         (c_THFloatTensor_stdall t biased) `shouldSatisfy` (\x -> and [(x < 0.3), (x > 0.2)])
--         c_THFloatTensor_free t
--   where
--     biased = 0

testsDouble = do
  gen <- c_THGenerator_new
  hspec $ do
    describe "random vectors" $ do
      it "uniform random is < bound" $ do
        t <- fromJust $ tensorNew [1000]
        c_THDoubleTensor_uniform t gen (-1.0) (1.0)
        (c_THDoubleTensor_maxall t) `shouldSatisfy` (< 1.001)
        c_THDoubleTensor_free t
      it "uniform random is > bound" $ do
        t <- fromJust $ tensorNew [1000]
        c_THDoubleTensor_uniform t gen (-1.0) (1.0)
        (c_THDoubleTensor_maxall t) `shouldSatisfy` (> (-1.001))
        c_THDoubleTensor_free t
      it "normal mean follows law of large numbers" $ do
        t <- fromJust $ tensorNew [100000]
        c_THDoubleTensor_normal t gen 1.55 0.25
        (c_THDoubleTensor_meanall t) `shouldSatisfy` (\x -> and [(x < 1.6), (x > 1.5)])
        c_THDoubleTensor_free t
      it "normal std follows law of large numbers" $ do
        t <- fromJust $ tensorNew [100000]
        c_THDoubleTensor_normal t gen 1.55 0.25
        (c_THDoubleTensor_stdall t biased) `shouldSatisfy` (\x -> and [(x < 0.3), (x > 0.2)])
        c_THDoubleTensor_free t
  where
    biased = 0

testsRandom :: IO ()
testsRandom = do
  testsDouble
  -- testsFloat

