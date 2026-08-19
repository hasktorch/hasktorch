{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.IndexSpec (spec) where

import Data.Proxy
import Lens.Family ((&), (.~), (^.), (%~))
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as D
import Torch.Typed.Factories (zeros)
import Torch.Index (slice)
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
  describe "getSlice" $ do
    it "selects with SliceAt, dropping the dimension" $ do
      let s = getSlice @'[SliceAt 1] t
      shape s `shouldBe` [3, 4]
      elems s `shouldBe` [12 .. 23]
    it "reaches inner dimensions through SliceAll" $ do
      let s = getSlice @'[SliceAll, SliceAt 0] t
      shape s `shouldBe` [2, 4]
      elems s `shouldBe` [0, 1, 2, 3, 12, 13, 14, 15]
    it "inserts a dimension with NewAxis" $ do
      shape (getSlice @'[NewAxis] t) `shouldBe` [1, 2, 3, 4]
    it "slices ranges and steps" $ do
      let s = getSlice @'[SliceFromUpTo 1 2, SliceUpTo 2, SliceFromWithStep 1 2] t
      shape s `shouldBe` [1, 2, 2]
      elems s `shouldBe` [13, 15, 17, 19]
    it "computes step lengths by ceiling division" $ do
      let s = getSlice @'[SliceAll, SliceFromUpToWithStep 0 3 2] t
      shape s `shouldBe` [2, 2, 4]
      elems s `shouldBe` ([0 .. 3] ++ [8 .. 11] ++ [12 .. 15] ++ [20 .. 23])
    it "accepts PyTorch-style syntax via the slice quasiquoter in type position" $ do
      elems (getSlice @[slice| 1 |] t) `shouldBe` elems (getSlice @'[SliceAt 1] t)
      elems (getSlice @[slice| :, 0 |] t) `shouldBe` elems (getSlice @'[SliceAll, SliceAt 0] t)
      shape (getSlice @[slice| None, :, 1:3 |] t) `shouldBe` [1, 2, 2, 4]
      elems (getSlice @[slice| 1:2, :2, 1::2 |] t) `shouldBe` [13, 15, 17, 19]
      shape (getSlice @[slice| :, ::2 |] t) `shouldBe` [2, 2, 4]
  describe "sliceLens" $ do
    it "reads, writes and modifies through the lens" $ do
      elems (t ^. sliceLens @'[SliceAt 1]) `shouldBe` [12 .. 23]
      elems (getSlice @'[SliceAt 0] (t & sliceLens @[slice| 0 |] .~ zeros)) `shouldBe` replicate 12 0
      let u = t & sliceLens @[slice| :, 1 |] %~ (* 100)
      elems (getSlice @[slice| :, 1 |] u) `shouldBe` map (* 100) ([4 .. 7] ++ [16 .. 19])
      elems (getSlice @[slice| :, 0 |] u) `shouldBe` ([0 .. 3] ++ [12 .. 15])
  describe "setSlice" $ do
    it "replaces the indexed part and leaves the rest" $ do
      let u = setSlice @'[SliceAt 0] t zeros
      elems (getSlice @'[SliceAt 0] u) `shouldBe` replicate 12 0
      elems (getSlice @'[SliceAt 1] u) `shouldBe` [12 .. 23]
