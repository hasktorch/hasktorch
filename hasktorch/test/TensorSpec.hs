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
    \(NonEmpty (x :: [Word8])) -> asValue (asTensor x) `shouldBe` x
  it "TensorLike [Int8]" $ property $
    \(NonEmpty (x :: [Int8])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Int16]" $ property $
    \(NonEmpty (x :: [Int16])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Int32]" $ property $
    \(NonEmpty (x :: [Int32])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Int]" $ property $
    \(NonEmpty (x :: [Int])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Int64]" $ property $
    \(NonEmpty (x :: [Int64])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Float]" $ property $
    \(NonEmpty (x :: [Float])) -> asValue (asTensor x)  `shouldBe` x
  it "TensorLike [Double]" $ property $
    \(NonEmpty (x :: [Double])) -> asValue (asTensor x)  `shouldBe` x

  it "TensorLike [Word8]" $ property $
    \(NonEmpty (x :: [Word8])) -> toDouble (select (asTensor x) 0 0) `shouldBe` fromIntegral (head x)
  it "TensorLike [Int8]" $ property $
    \(NonEmpty (x :: [Int8])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` fromIntegral (head x)
  it "TensorLike [Int16]" $ property $
    \(NonEmpty (x :: [Int16])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` fromIntegral (head x)
  it "TensorLike [Int32]" $ property $
    \(NonEmpty (x :: [Int32])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` fromIntegral (head x)
  it "TensorLike [Int]" $ property $
    \(NonEmpty (x :: [Int])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` fromIntegral (head x)
  it "TensorLike [Int64]" $ property $
    \(NonEmpty (x :: [Int64])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` fromIntegral (head x)
  it "TensorLike [Float]" $ property $
    \(NonEmpty (x :: [Float])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` realToFrac (head x)
  it "TensorLike [Double]" $ property $
    \(NonEmpty (x :: [Double])) -> toDouble (select (asTensor x) 0 0)  `shouldBe` realToFrac (head x)

  it "length of TensorLike [Word8]" $ property $
    \(NonEmpty (x :: [Word8])) -> shape (asTensor x) `shouldBe` [length x]
  it "length of TensorLike [Int8]" $ property $
    \(NonEmpty (x :: [Int8])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Int16]" $ property $
    \(NonEmpty (x :: [Int16])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Int32]" $ property $
    \(NonEmpty (x :: [Int32])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Int]" $ property $
    \(NonEmpty (x :: [Int])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Int64]" $ property $
    \(NonEmpty (x :: [Int64])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Float]" $ property $
    \(NonEmpty (x :: [Float])) -> shape (asTensor x)  `shouldBe` [length x]
  it "length of TensorLike [Double]" $ property $
    \(NonEmpty (x :: [Double])) -> shape (asTensor x)  `shouldBe` [length x]
