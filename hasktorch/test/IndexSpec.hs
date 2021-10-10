{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

module IndexSpec (spec) where

import Control.Arrow ((&&&))
import Lens.Family
import Test.Hspec
import Test.QuickCheck
import Torch.DType
import Torch.Index
import Torch.Lens
import Torch.Tensor
import Torch.TensorFactories

spec :: Spec
spec = do
  describe "slice" $ do
    it "None" $ do
      [slice|None|] `shouldBe` None
    it "Ellipsis" $ do
      [slice|Ellipsis|] `shouldBe` Ellipsis
    it "..." $ do
      [slice|...|] `shouldBe` Ellipsis
    it "123" $ do
      [slice|123|] `shouldBe` 123
    it "-123" $ do
      [slice|-123|] `shouldBe` -123
    it "True" $ do
      [slice|True|] `shouldBe` True
    it "False" $ do
      [slice|False|] `shouldBe` False
    it ":" $ do
      [slice|:|] `shouldBe` Slice ()
    it "::" $ do
      [slice|::|] `shouldBe` Slice ()
    it "1:" $ do
      [slice|1:|] `shouldBe` Slice (1, None)
    it "1::" $ do
      [slice|1::|] `shouldBe` Slice (1, None)
    it ":3" $ do
      [slice|:3|] `shouldBe` Slice (None, 3)
    it ":3:" $ do
      [slice|:3:|] `shouldBe` Slice (None, 3)
    it "::2" $ do
      [slice|::2|] `shouldBe` Slice (None, None, 2)
    it "1:3" $ do
      [slice|1:3|] `shouldBe` Slice (1, 3)
    it "1::2" $ do
      [slice|1::2|] `shouldBe` Slice (1, None, 2)
    it ":3:2" $ do
      [slice|:3:2|] `shouldBe` Slice (None, 3, 2)
    it "1:3:2" $ do
      [slice|1:3:2|] `shouldBe` Slice (1, 3, 2)
    it "1,2,3" $ do
      [slice|1,2,3|] `shouldBe` (1, 2, 3)
    it "1 , 2, 3" $ do
      [slice|1 , 2, 3|] `shouldBe` (1, 2, 3)
    it "1 , 2, 3" $ do
      let i = 1
      [slice|{i} , 2, 3|] `shouldBe` (1, 2, 3)
  describe "indexing" $ do
    it "pick up a value" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ! [slice|1,0,2|]
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([], 8 :: Int))
    it "intercalate" $ do
      let x = zeros' [6]
          i = [slice|0::2|]
      (dtype &&& shape &&& asValue) (maskedFill x i (arange' 1 4 1)) `shouldBe` (Float, ([6], [1, 0, 2, 0, 3, 0] :: [Float]))
    it "negative index" $ do
      let x = arange' 1 5 1
          i = [slice|-1|]
      (dtype &&& shape &&& asValue) (x ! i) `shouldBe` (Float, ([], 4 :: Float))
  describe "indexing with lens" $ do
    it "pick up a value" $ do
      let x = asTensor ([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]] :: [[[Int]]])
          r = x ^. [lslice|1,0,2|]
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([], 8 :: Int))
    it "intercalate" $ do
      let x = zeros' [6]
          i = [lslice|0::2|] :: Lens' Tensor Tensor
      (dtype &&& shape &&& asValue) (x & i .~ arange' 1 4 1) `shouldBe` (Float, ([6], [1, 0, 2, 0, 3, 0] :: [Float]))
    it "negative index" $ do
      let x = arange' 1 5 1
          i = [lslice|-1|]
      (dtype &&& shape &&& asValue) (x ^. i) `shouldBe` (Float, ([], 4 :: Float))
