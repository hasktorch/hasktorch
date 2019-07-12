{-# LANGUAGE ScopedTypeVariables #-}

module TensorSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Control.Exception.Safe

import Torch.Tensor
import Torch.DType
import Torch.TensorFactories
import Torch.Functions
import Torch.TensorOptions
import Data.Word
import Data.Int

spec :: Spec
spec = do
  it "TensorLike Word8" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Word8)
  it "TensorLike Int8" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Int8)
  it "TensorLike Int16" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Int16)
  it "TensorLike Int32" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Int32)
  it "TensorLike Int" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Int)
  it "TensorLike Int64" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Int64)
  it "TensorLike Float" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Float)
  it "TensorLike Double" $ property $
    \x -> asValue (asTensor x) `shouldBe` (x :: Double)

  it "TensorLike [Word8]" $ property $
    \(NonEmpty (x :: [Word8])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Int8]" $ property $
    \(NonEmpty (x :: [Int8])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Int16]" $ property $
    \(NonEmpty (x :: [Int16])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Int32]" $ property $
    \(NonEmpty (x :: [Int32])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Int]" $ property $
    \(NonEmpty (x :: [Int])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Int64]" $ property $
    \(NonEmpty (x :: [Int64])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Float]" $ property $
    \(NonEmpty (x :: [Float])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` realToFrac (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx
  it "TensorLike [Double]" $ property $
    \(NonEmpty (x :: [Double])) -> do
      asValue (asTensor x) `shouldBe` x
      toDouble (select (asTensor x) 0 0) `shouldBe` realToFrac (head x)
      shape (asTensor x) `shouldBe` [length x]
      let xx = replicate 5 x
      asValue (asTensor xx) `shouldBe` xx
      let xxx = replicate 3 xx
      asValue (asTensor xxx) `shouldBe` xxx

  it "invalid cast of TensorLike a" $ do
    let x = asTensor (10 :: Int)
    (dtype x) `shouldBe` Int64
    (print (asValue x :: Double)) `shouldThrow` anyException
  it "invalid cast of TensorLike [a]" $ do
    let x = asTensor ([0..10] :: [Int])
    (print (asValue x :: [Double])) `shouldThrow` anyException

  it "lists having different length" $ do
    (print (asTensor ([[1],[1,2]] :: [[Double]]))) `shouldThrow` anyException
