{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.Dynamic.TensorSpec where

import Test.Hspec
import Test.Hspec.QuickCheck hiding (resize)
import Test.QuickCheck ((.||.), (==>), (===), property)
import Test.QuickCheck.Monadic
import Data.List (genericLength)
import GHC.Exts

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  prop "fromList / toList forms the identity" $ \xs ->
    toList ((fromList xs) :: Dynamic) `shouldBe` xs

  describe "empty" $ do
    it "has 0 dimensions" $ nDimension empty `shouldBe` 0
    it "has 0 elements" $ nElement empty `shouldBe` 0
    it "is contiguous" $ isContiguous empty `shouldBe` True
    it "is the same size as other empty tensors" $ isSameSizeAs empty empty `shouldBe` True
    it "is 'set to' the same storage as other empty tensors (ie: is set to NULL)" $ isSetTo empty empty `shouldBe` True

  describe "fromList" $ do
    prop "will have 1 dimension (given non-empty list)" $ \xs -> not (null xs) ==> nDimension (fromList xs) === 1
    prop "will have the same number of elements as input list" $ \xs -> nElement (fromList xs) === genericLength xs
    prop "will be contiguous" $ \xs -> isContiguous (fromList xs) === True

  describe "operations on vectors (made fromList)" $ do
    describe "newClone on vectors" $ do
      prop "will have identical dimensions" $ \xs -> let v = fromList xs in nDimension (newClone v) === nDimension v
      prop "will have identical elements" $ \xs -> let v = fromList xs in nElement (newClone v) === nElement v
      prop "will both identically contiguous" $ \xs -> let v = fromList xs in isContiguous (newClone v) === isContiguous v
      prop "will not be set to the same storage" $ \xs -> let v = fromList xs in isSetTo (newClone v) v === False

    prop "get1d will be Nothing for empty tensors and the index'd value otherwise" $ \xs i ->
      let
        t = fromList xs
        i' = fromIntegral i
      in     (length xs < i' ==> get1d t i === Nothing)
        .||. (i' < length xs ==> get1d t i === Just (xs !! i'))

    prop "get2d will always be Nothing" $ \xs i0 i1       -> get2d (fromList xs) i0 i1       === Nothing
    prop "get3d will always be Nothing" $ \xs i0 i1 i2    -> get3d (fromList xs) i0 i1 i2    === Nothing
    prop "get4d will always be Nothing" $ \xs i0 i1 i2 i3 -> get4d (fromList xs) i0 i1 i2 i3 === Nothing

  it "size" $ pending
  it "sizeDesc" $ pending
  it "newClone on higher-ranked tensor" $ pending
  it "get1d on higher-ranked tensor" $ pending
  it "get2d on higher-ranked tensor" $ pending
  it "get3d on higher-ranked tensor" $ pending
  it "get4d on higher-ranked tensor" $ pending
  it "isContiguous on higher-ranked tensor" $ pending
  it "isSameSizeAs on higher-ranked tensor" $ pending
  it "isSetTo on higher-ranked tensor" $ pending
  it "isSize" $ pending
  it "_narrow" $ pending
  it "newExpand" $ pending
  it "_expand" $ pending
  it "_expandNd" $ pending
  it "newContiguous" $ pending
  it "newNarrow" $ pending
  it "newSelect" $ pending
  it "newSizeOf" $ pending
  it "newStrideOf" $ pending
  it "newTranspose" $ pending
  it "newUnfold" $ pending
  it "newView" $ pending
  it "newWithSize" $ pending
  it "newWithSize1d" $ pending
  it "newWithSize2d" $ pending
  it "newWithSize3d" $ pending
  it "newWithSize4d" $ pending
  it "newWithStorage" $ pending
  it "newWithStorage1d" $ pending
  it "newWithStorage2d" $ pending
  it "newWithStorage3d" $ pending
  it "newWithStorage4d" $ pending
  it "newWithTensor" $ pending
  it "_resize" $ pending
  it "_resize1d" $ pending
  it "_resize2d" $ pending
  it "_resize3d" $ pending
  it "_resize4d" $ pending
  it "_resize5d" $ pending
  it "_resizeAs" $ pending
  it "resizeAs" $ pending
  it "_resizeNd" $ pending
  it "retain" $ pending
  it "_select" $ pending
  it "_set" $ pending
  it "_set1d" $ pending
  it "_set2d" $ pending
  it "_set3d" $ pending
  it "_set4d" $ pending
  it "_setStorage" $ pending
  it "_setStorage1d" $ pending
  it "_setStorage2d" $ pending
  it "_setStorage3d" $ pending
  it "_setStorage4d" $ pending
  it "_setStorageNd" $ pending
  it "_squeeze" $ pending
  it "_squeeze1d" $ pending
  it "storage" $ pending
  it "storageOffset" $ pending
  it "stride" $ pending
  it "_transpose" $ pending
  it "_unfold" $ pending
  it "_unsqueeze1d" $ pending
  it "vector" $ pending
  it "matrix" $ pending
  it "cuboid" $ pending
  it "hyper" $ pending
  it "tensorSlices" $ pending

