{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorSpec (spec) where

import Control.Arrow ((&&&))
import Control.Exception.Safe
import Numeric.Half
import Data.Complex
import Data.Int
import Data.Word
import Test.Hspec
import Test.QuickCheck
import Torch.DType
import Torch.Functional
import Torch.Tensor
import Torch.TensorFactories
import Torch.TensorOptions
import Test.QuickCheck.Arbitrary

instance Arbitrary Half where
  arbitrary = arbitrarySizedFractional
  shrink    = shrinkDecimal

spec :: Spec
spec = do
  describe "TensorLike" $ do
    it "TensorLike Bool" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Bool)
    it "TensorLike Word8" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Word8)
    it "TensorLike Int8" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Int8)
    it "TensorLike Int16" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Int16)
    it "TensorLike Int32" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Int32)
    it "TensorLike Int" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Int)
    it "TensorLike Int64" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Int64)
    it "TensorLike Half" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Half)
    it "TensorLike Float" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Float)
    it "TensorLike Double" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Double)
    it "TensorLike ComplexHalf" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Complex Half)
    it "TensorLike Complex Float" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Complex Float)
    it "TensorLike Complex Double" $
      property $
        \x -> asValue (asTensor x) `shouldBe` (x :: Complex Double)

    it "Compare internal expression of c++ with Storable expression of haskell" $ do
      show (asTensor [True, False, True, False])
        `shouldBe` "Tensor Bool [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Word8]))
        `shouldBe` "Tensor UInt8 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Int8]))
        `shouldBe` "Tensor Int8 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Int16]))
        `shouldBe` "Tensor Int16 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Int32]))
        `shouldBe` "Tensor Int32 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Int]))
        `shouldBe` "Tensor Int64 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Int64]))
        `shouldBe` "Tensor Int64 [4] [ 1,  0,  1,  0]"
      show (asTensor ([1, 0, 1, 0] :: [Float]))
        `shouldBe` "Tensor Float [4] [ 1.0000   ,  0.0000,  1.0000   ,  0.0000]"
      show (asTensor ([1, 0, 1, 0] :: [Double]))
        `shouldBe` "Tensor Double [4] [ 1.0000   ,  0.0000,  1.0000   ,  0.0000]"
      show (asTensor ([[]] :: [[Int]]))
        `shouldBe` "Tensor Int64 [1,0] [[]]"
      show (asTensor ([[1]] :: [[Int]]))
        `shouldBe` "Tensor Int64 [1,1] [[ 1]]"

    it "TensorLike [Bool]" $
      property $
        \(NonEmpty (x :: [Bool])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` (if (head x) then 1 else 0)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Word8]" $
      property $
        \(NonEmpty (x :: [Word8])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Int8]" $
      property $
        \(NonEmpty (x :: [Int8])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Int16]" $
      property $
        \(NonEmpty (x :: [Int16])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Int32]" $
      property $
        \(NonEmpty (x :: [Int32])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Int]" $
      property $
        \(NonEmpty (x :: [Int])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Int64]" $
      property $
        \(NonEmpty (x :: [Int64])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` fromIntegral (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Half]" $
      property $
        \(NonEmpty (x :: [Half])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` realToFrac (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Float]" $
      property $
        \(NonEmpty (x :: [Float])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` realToFrac (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Double]" $
      property $
        \(NonEmpty (x :: [Double])) -> do
          asValue (asTensor x) `shouldBe` x
          toDouble (select 0 0 (asTensor x)) `shouldBe` realToFrac (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Complex Half]" $
      property $
        \(NonEmpty (x :: [Complex Half])) -> do
          asValue (asTensor x) `shouldBe` x
          asValue (select 0 0 (asTensor x)) `shouldBe` (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Complex Float]" $
      property $
        \(NonEmpty (x :: [Complex Float])) -> do
          asValue (asTensor x) `shouldBe` x
          asValue (select 0 0 (asTensor x)) `shouldBe` (head x)
          shape (asTensor x) `shouldBe` [length x]
          let xx = replicate 5 x
          asValue (asTensor xx) `shouldBe` xx
          let xxx = replicate 3 xx
          asValue (asTensor xxx) `shouldBe` xxx
    it "TensorLike [Complex Double]" $
      property $
        \(NonEmpty (x :: [Complex Double])) -> do
          asValue (asTensor x) `shouldBe` x
          asValue (select 0 0 (asTensor x)) `shouldBe` (head x)
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
      let x = asTensor ([0 .. 10] :: [Int])
      (print (asValue x :: [Double])) `shouldThrow` anyException

    it "lists having different length" $ do
      (print (asTensor ([[1], [1, 2]] :: [[Double]]))) `shouldThrow` anyException
    it "cast of Tensor" $ do
      let x = asTensor ([0 .. 10] :: [Int])
      (dtype (toType Float x)) `shouldBe` Float

  describe "indexing" $ do
    it "pick up a value" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! (1, 0, 2)
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([], 8 :: Int))
    it "pick up a bottom tensor" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! (1, 0)
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([3], [6 :: Int, 7, 8]))
    it "make a slice of bottom values" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! ((), (), 1)
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 2], [[1 :: Int, 4], [7, 10]]))
    it "ellipsis" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! (Ellipsis, 1)
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 2], [[1 :: Int, 4], [7, 10]]))
    it "make a slice via muliple slices" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! ((), (Slice (1, None)))
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 1, 3], [[[3 :: Int, 4, 5]], [[9, 10, 11]]]))
    it "make a slice via muliple slices" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! ((), (Slice (1, None)), (Slice (0, 1)))
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 1, 1], [[[3 :: Int]], [[9]]]))
  describe "masked fill" $ do
    it "Fill a value" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = maskedFill x (1, 0, 2) (9 :: Int)
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 2, 3], [[[0, 1, 2], [3, 4, 5]], [[6, 7, 9], [9, 10, 11]]] :: [[[Int]]]))
    it "Fill a bottom tensor" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = maskedFill x (1, 0) [8 :: Int, 8, 8]
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 2, 3], [[[0, 1, 2], [3, 4, 5]], [[8, 8, 8], [9, 10, 11]]] :: [[[Int]]]))
    it "masked fill by boolean" $ do
      let m = asTensor ([[[True, True, False], [False, False, False]], [[True, False, False], [False, False, False]]] :: [[[Bool]]])
          x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = maskedFill x m [8 :: Int, 8, 8]
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([2, 2, 3], [[[8, 8, 2], [3, 4, 5]], [[8, 7, 8], [9, 10, 11]]] :: [[[Int]]]))
