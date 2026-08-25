{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE NoStarIsType #-}

module Torch.Typed.BufferSpec (spec) where

import Data.Maybe (fromJust)
import Test.Hspec
import qualified Torch.DType as D
import qualified Torch.Device as D
import qualified Torch.Functional as F
import qualified Torch.Tensor as UT
import Torch.Typed
import Prelude hiding (exp, sum)

type Dev = '(D.CPU, 0)

row :: [Float] -> Tensor Dev 'D.Float '[2]
row = UnsafeMkTensor . UT.asTensor

elems :: Tensor Dev 'D.Float sh -> [Float]
elems = UT.asValue . UT.reshape [-1] . toDynamic

spec :: Spec
spec = describe "Buffer (a KV-cache-shaped bounded buffer)" $ do
  let b0 = emptyBuffer :: Buffer Dev 'D.Float 4 '[2]
      b3 = fromJust (append (row [5, 6]) =<< append (row [3, 4]) =<< append (row [1, 2]) b0)
  it "appends into the static backing tensor, zeros after the prefix" $ do
    used b3 `shouldBe` 3
    capacity b3 `shouldBe` 4
    elems (bufferTensor b3) `shouldBe` [1, 2, 3, 4, 5, 6, 0, 0]
  it "refuses to overfill" $ do
    let b4 = fromJust (append (row [7, 8]) b3)
    used b4 `shouldBe` 4
    case append (row [9, 9]) b4 of
      Nothing -> pure ()
      Just _ -> expectationFailure "appended past capacity"
  it "masks the empty tail" $ do
    (UT.asValue (toDynamic (validMask b3)) :: [Bool])
      `shouldBe` [True, True, True, False]
  it "masked attention over the buffer equals attention over the true prefix" $ do
    -- scores against every slot, then softmax with the additive mask: the
    -- empty slots must get exactly zero weight, so the result matches the
    -- softmax computed over only the filled prefix.
    let q = toDynamic (row [1, 0.5])
        scores = F.matmul (toDynamic (bufferTensor b3)) q -- [4]
        masked = scores + toDynamic (attentionMask b3)
        ws = UT.asValue (F.softmax (F.Dim 0) masked) :: [Float]
        prefixScores = take 3 (UT.asValue scores :: [Float])
        prefix = UT.asValue (F.softmax (F.Dim 0) (UT.asTensor prefixScores)) :: [Float]
    maximum (zipWith (\a b -> Prelude.abs (a - b)) (take 3 ws) prefix)
      `shouldSatisfy` (< 1e-6)
    drop 3 ws `shouldBe` [0]
