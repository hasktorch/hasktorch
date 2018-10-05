{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Indef.StorageSpec where

import Control.Monad
import GHC.Int
import Test.Hspec
import Test.Hspec.QuickCheck hiding (resize)
import Test.QuickCheck ((.&&.), (==>), (===), property)
import Test.QuickCheck.Monadic
import GHC.Exts
import Debug.Trace
import Control.Monad.IO.Class

import Torch.Indef.Storage

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  prop "fromList / toList should form the identity" $ \xs ->
    toList (fromList xs :: Storage) `shouldBe` xs

  prop "size should be the identical to the length" $ \xs ->
    size (fromList xs) `shouldBe` (length xs)

  describe "set / get" $ do
    it "can update any index" $ property $ \ (xs, x, i) ->
      length xs > 1 ==>
      i >= 0 && i < length xs ==>
      let st = (fromList xs)
      in get st i /= x ==> do
          set st i x
          get st i `shouldBe` x

  describe "empty" $ do
    it "has a size of 0" $ do
      let st = empty
      size empty `shouldBe` 0

  prop "newWithSize returns a size of input argument" $ \i ->
    i >= 0 ==> size (newWithSize i) `shouldBe` i

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

  xdescribe "resize" $ do
    prop "will truncate storage if the new size is less than the old one (old size <100)" $ \xs i ->
      length xs < 100 && length xs > fromIntegral i ==> do
        let st = fromList xs
        resize st i
        size st `shouldBe` fromIntegral i
        toList st `shouldBe` take (fromIntegral i) xs

    prop "will expand storage if the new size is greater than the old one (new size <100)" $ \xs i ->
      length xs < fromIntegral i && i < 100 ==> do
        let st = fromList xs
        resize st i
        size st `shouldBe` fromIntegral i
        take (length xs) (toList st) `shouldBe` xs


