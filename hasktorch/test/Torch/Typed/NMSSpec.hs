{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Tests for the non-maximum suppression in "Torch.Typed.Vision".
--
-- The central check: 'iou' is one polymorphic formula, and its two
-- instantiations — @a = Tensor@ (the vectorized 'boxIou') and @a = Float@
-- (the per-element reference below) — must agree /exactly/.  The reference
-- implementation is not hand-written a second time; it is the same code.
module Torch.Typed.NMSSpec (spec) where

import Data.Vector.Sized (Vector)
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Tensor as D
import Torch.Typed.Representable
import Torch.Typed.Tensor
import Torch.Typed.Vision (Box (..), boxIou, iou, nms)

-- five detections: 1 overlaps 0, 3 sits inside 2, 4 stands alone
-- columns: x1, y1, x2, y2, score
boxData :: [[Float]]
boxData =
  [ [0, 0, 10, 10, 0.9],
    [1, 1, 11, 11, 0.8],
    [20, 20, 30, 30, 0.7],
    [21, 21, 29, 29, 0.6],
    [50, 50, 60, 60, 0.5]
  ]

dets5 :: NamedTensor '(D.CPU, 0) 'D.Float '[Vector 5, Box]
dets5 = tabulateList @'[Vector 5, Box] @'D.Float @'(D.CPU, 0) (\[i, k] -> boxData !! i !! k)

-- the same iou formula at a = Float, one pair at a time
boxAt :: Int -> Box Float
boxAt i = Box (at 0) (at 1) (at 2) (at 3) (at 4)
  where
    at k = indexList dets5 [i, k]

iouMat :: [[Float]]
iouMat = D.asValue (toDynamic (boxIou dets5))

spec :: Spec
spec = do
  describe "boxIou" $ do
    it "the Tensor instantiation of iou agrees exactly with the Float one" $ do
      iouMat `shouldBe` [[iou (boxAt i) (boxAt j) | j <- [0 .. 4]] | i <- [0 .. 4]]
    it "is 1 on the diagonal and symmetric" $ do
      [iouMat !! i !! i | i <- [0 .. 4]] `shouldBe` [1, 1, 1, 1, 1]
      [iouMat !! i !! j | i <- [0 .. 4], j <- [0 .. 4]]
        `shouldBe` [iouMat !! j !! i | i <- [0 .. 4], j <- [0 .. 4]]
  describe "nms" $ do
    it "keeps the best box of each cluster" $ do
      map fromEnum (nms 0.5 dets5) `shouldBe` [0, 2, 4]
    it "keeps everything when the threshold admits all overlap" $ do
      map fromEnum (nms 1.0 dets5) `shouldBe` [0, 1, 2, 3, 4]
    it "keeps only non-overlapping boxes when the threshold is zero" $ do
      map fromEnum (nms 0.0 dets5) `shouldBe` [0, 2, 4]
