{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.IndexSpec (spec) where

import Data.Proxy
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as D
import Torch.Typed.Factories (zeros)
import Torch.Typed.Index
import Torch.Typed.Tensor

-- t[i][j][k] = i*12 + j*4 + k
t :: Tensor '(D.CPU, 0) 'D.Float '[2, 3, 4]
t = UnsafeMkTensor (D.reshape [2, 3, 4] (D.asTensor [0 .. 23 :: Float]))

elems :: Tensor '(D.CPU, 0) 'D.Float sh -> [Float]
elems = D.asValue . D.reshape [-1] . toDynamic

-- compile-time checks of the shape computation
testAt :: Proxy (IndexedShape '[SliceAt 1] '[2, 3, 4]) -> Proxy '[3, 4]
testAt = id

testNewAxis :: Proxy (IndexedShape '[NewAxis, SliceAll] '[2, 3]) -> Proxy '[1, 2, 3]
testNewAxis = id

testSteps :: Proxy (IndexedShape '[SliceFromUpToWithStep 0 3 2] '[3]) -> Proxy '[2]
testSteps = id

testTrailing :: Proxy (IndexedShape '[SliceAt 0] '[2, 3, 4]) -> Proxy '[3, 4]
testTrailing = id

spec :: Spec
spec = do
  describe "slice" $ do
    it "selects with SliceAt, dropping the dimension" $ do
      let s = slice @'[SliceAt 1] t
      shape s `shouldBe` [3, 4]
      elems s `shouldBe` [12 .. 23]
    it "reaches inner dimensions through SliceAll" $ do
      let s = slice @'[SliceAll, SliceAt 0] t
      shape s `shouldBe` [2, 4]
      elems s `shouldBe` [0, 1, 2, 3, 12, 13, 14, 15]
    it "inserts a dimension with NewAxis" $ do
      shape (slice @'[NewAxis] t) `shouldBe` [1, 2, 3, 4]
    it "slices ranges and steps" $ do
      let s = slice @'[SliceFromUpTo 1 2, SliceUpTo 2, SliceFromWithStep 1 2] t
      shape s `shouldBe` [1, 2, 2]
      elems s `shouldBe` [13, 15, 17, 19]
    it "computes step lengths by ceiling division" $ do
      let s = slice @'[SliceAll, SliceFromUpToWithStep 0 3 2] t
      shape s `shouldBe` [2, 2, 4]
      elems s `shouldBe` ([0 .. 3] ++ [8 .. 11] ++ [12 .. 15] ++ [20 .. 23])
  describe "setSlice" $ do
    it "replaces the indexed part and leaves the rest" $ do
      let u = setSlice @'[SliceAt 0] t zeros
      elems (slice @'[SliceAt 0] u) `shouldBe` replicate 12 0
      elems (slice @'[SliceAt 1] u) `shouldBe` [12 .. 23]
