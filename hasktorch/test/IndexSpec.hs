{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}


module IndexSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Torch.Tensor
import Torch.Index
import Control.Arrow                  ( (&&&) )
import Torch.DType
import Torch.TensorFactories

spec :: Spec
spec = do
  describe "pySlice" $ do
    it "None" $ do
      [pySlice|None|] `shouldBe` None
    it "Ellipsis" $ do
      [pySlice|Ellipsis|] `shouldBe` Ellipsis
    it "..." $ do
      [pySlice|...|] `shouldBe` Ellipsis
    it "123" $ do
      [pySlice|123|] `shouldBe` 123
    it "True" $ do
      [pySlice|True|] `shouldBe` True
    it "False" $ do
      [pySlice|False|] `shouldBe` False
    it ":" $ do
      [pySlice|:|] `shouldBe` Slice ()
    it "::" $ do
      [pySlice|::|] `shouldBe` Slice ()
    it "1:" $ do
      [pySlice|1:|] `shouldBe` Slice (1, None)
    it "1::" $ do
      [pySlice|1::|] `shouldBe` Slice (1, None)
    it ":3" $ do
      [pySlice|:3|] `shouldBe` Slice (None, 3)
    it ":3:" $ do
      [pySlice|:3:|] `shouldBe` Slice (None, 3)
    it "::2" $ do
      [pySlice|::2|] `shouldBe` Slice (None, None, 2)
    it "1:3" $ do
      [pySlice|1:3|] `shouldBe` Slice (1, 3)
    it "1::2" $ do
      [pySlice|1::2|] `shouldBe` Slice (1, None, 2)
    it ":3:2" $ do
      [pySlice|:3:2|] `shouldBe` Slice (None, 3, 2)
    it "1:3:2" $ do
      [pySlice|1:3:2|] `shouldBe` Slice (1, 3, 2)
    it "1,2,3" $ do
      [pySlice|1,2,3|] `shouldBe` (1, 2, 3)
    it "1 , 2, 3" $ do
      [pySlice|1 , 2, 3|] `shouldBe` (1, 2, 3)
  describe "indexing" $ do
    it "pick up a value" $ do
      let x = asTensor ([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]] :: [[[Int]]])
          r = x ! [pySlice|1,0,2|]
      (dtype &&& shape &&& asValue) r `shouldBe` (Int64, ([], 8 :: Int))
    it "intercalate" $ do
      let x = zeros' [6]
          i = [pySlice|0::2|]
      (dtype &&& shape &&& asValue) (maskedFill x i (arange' 1 4 1)) `shouldBe` (Float, ([6], [1,0,2,0,3,0] :: [Float]))
