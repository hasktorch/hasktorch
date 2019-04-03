{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.StorageSpec where

import Control.Monad
import GHC.Int
import Test.Hspec
import Test.Hspec.QuickCheck hiding (resize)
import Test.QuickCheck ((.&&.), (==>), (===), property, Arbitrary(..))
import qualified Test.QuickCheck as QC
import Test.QuickCheck.Monadic
import GHC.Exts
import Debug.Trace
import Control.Monad.IO.Class
import Data.List
import Data.Word
import GHC.Natural

import Torch.Indef.Storage

data ListWithIx a = ListWithIx Word [a]
  deriving (Eq, Show)

instance Arbitrary a => Arbitrary (ListWithIx a) where
  arbitrary = do
    xs <- QC.listOf1 arbitrary
    ix <- QC.choose (0, genericLength xs - 1)
    pure $ ListWithIx ix xs

data ListWithOOB a = ListWithOOB Word [a]
  deriving (Eq, Show)

instance Arbitrary a => Arbitrary (ListWithOOB a) where
  arbitrary = do
    xs <- QC.listOf1 arbitrary
    ix <- QC.choose (genericLength xs, genericLength xs * 5)
    pure $ ListWithOOB ix xs

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "fromList / storagedata" $ do
    let
      xs = [0..500]
      st = fromList xs

    it "is instantiated with an offset of 0" $
      size st `shouldBe` length xs

    it "returns xs for storagedata" $
      storagedata st `shouldBe` xs

    it ("can access the first 50 elements by `get`") $ do
      forM_ [0..50] $ \i ->
        get st i `shouldBe` fromIntegral i

    it "should continue to be consistent after gets" $
      storagedata st `shouldBe` xs

  prop "fromList / toList should form the identity" $ \xs ->
    toList (fromList xs :: Storage) `shouldBe` xs

  prop "size should be the identical to the length" $ \xs ->
    size (fromList xs) `shouldBe` (length xs)

  describe "set / get" $ do
    it "can update any index" $ property $ \ (x, ListWithIx i xs) -> do
      let st = fromList xs
      get st (fromIntegral i) /= x ==> do
        set st (fromIntegral i) x
        get st (fromIntegral i) `shouldBe` x

  describe "empty" $ do
    it "has a size of 0" $ do
      let st = empty
      size empty `shouldBe` 0

  prop "newWithSize returns a size of input argument" $ \(i::Word16) ->
    size (newWithSize $ fromIntegral i) `shouldBe` fromIntegral i

  prop "newWithSize1 has size 1 and sets the argument" $ \v0 ->
    let st = newWithSize1 v0
    in size st === 1 .&&. get st 0 === v0

  prop "newWithSize2 has size 2 and sets the arguments" $ \v0 v1 ->
    let st = newWithSize2 v0 v1
    in   size st === 2
    .&&. get st 0 === v0
    .&&. get st 1 === v1

  prop "newWithSize3 has size 3 and sets the arguments" $ \v0 v1 v2 ->
    let st = newWithSize3 v0 v1 v2
    in   size st === 3
    .&&. get st 0 === v0
    .&&. get st 1 === v1
    .&&. get st 2 === v2

  prop "newWithSize4 has size 4 and sets the arguments" $ \v0 v1 v2 v3 ->
    let st = newWithSize4 v0 v1 v2 v3
    in   size st === 4
    .&&. get st 0 === v0
    .&&. get st 1 === v1
    .&&. get st 2 === v2
    .&&. get st 3 === v3

  prop "fill will overwrite the storage contents" $ \xs v ->
    length xs > 1 ==> do
    let st = fromList xs
    toList st `shouldBe` xs
    fill st v
    toList st `shouldSatisfy` (all (== v))

  describe "resize" $ do
    prop "will truncate storage if the new size is less than the old one (old size < 100)" $
      \(ListWithIx i xs) -> length xs < 100 ==> do
        let st = fromList xs
        resize st (fromIntegral i)
        toList st `shouldBe` take (fromIntegral i) xs

    prop "will expand storage if the new size is greater than the old one (new size <100)" $
      \(ListWithOOB i xs) -> i < 100 ==> do
        let st = fromList xs
        resize st (fromIntegral i)
        take (length xs) (toList st) `shouldBe` xs

