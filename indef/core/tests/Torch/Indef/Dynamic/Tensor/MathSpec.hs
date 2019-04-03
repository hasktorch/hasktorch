{-# LANGUAGE DataKinds #-}
module Torch.Indef.Dynamic.Tensor.MathSpec where

import Torch.Indef.Types
import Torch.Indef.Dynamic.Tensor
import Torch.Indef.Dynamic.Tensor.Math

import Control.Monad
import Numeric.Dimensions
import Test.Hspec
import Test.QuickCheck hiding (vector)
import qualified Test.QuickCheck as QC

newtype SomeDimList = SomeDimList [Word]
  deriving (Eq, Ord, Show)

instance Arbitrary SomeDimList where
  arbitrary = do
    len <- QC.choose (1, 3)
    ixs <- replicateM len $ QC.choose (1, 20)
    pure $ SomeDimList ixs

newtype Ord2Tuple = Ord2Tuple (Int, Int)
  deriving (Eq, Show)

instance Arbitrary Ord2Tuple where
  arbitrary = Ord2Tuple <$>
    suchThat
      ((,) <$> arbitrary <*> arbitrary)
      (\(l, u) -> l < u && u < 100 && l > -100)


instance Arbitrary Dynamic where
  arbitrary = do
    SomeDimList ds <- arbitrary
    pure $ new' (someDimsVal ds)

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "fill_" fill_Spec
  describe "zero_" zero_Spec
  describe "zeros_" zeros_Spec
  describe "zerosLike_" zerosLike_Spec
  describe "ones_" ones_Spec
  describe "onesLike_" onesLike_Spec
  describe "numel" numelSpec
  describe "arange" arangeSpec
  describe "cat" catSpec

fill_Spec :: Spec
fill_Spec = do
  it "fills inplace" $ property $ \ten i -> do
    fill_ ten i
    tensordata ten `shouldSatisfy` (all (== i))

zero_Spec :: Spec
zero_Spec = do
  it "zeros inplace" $ property $ \ten -> do
    fill_ ten 1
    zero_ ten
    tensordata ten `shouldSatisfy` (all (== 0))

zeros_Spec :: Spec
zeros_Spec =
  it "mutates a tensor, inplace, resizing the tensor to the given IndexStorage and filling with zeros"
    pending

zerosLike_Spec :: Spec
zerosLike_Spec =
  it "mutates a tensor, inplace, resizing the tensor to the given tensor and filling with zeros"
    pending

ones_Spec :: Spec
ones_Spec =
  it "mutates a tensor, inplace, resizing the tensor to the given IndexStorage and filling with ones"
    pending

onesLike_Spec :: Spec
onesLike_Spec =
  it "mutates a tensor, inplace, resizing the tensor to the given tensor and filling with ones"
    pending

numelSpec :: Spec
numelSpec =
  it "returns the number of elements in the tensor" . property $ \t ->
    fromIntegral (numel t) `shouldBe` product (shape t)

arangeSpec :: Spec
arangeSpec =
  it "returns the number of elements in the tensor" . property $ \(Ord2Tuple (l', u')) ->
    let
      l = fromIntegral l'
      u = fromIntegral u'
    in
      tensordata (arange l u 1) `shouldBe` [l..u - 1]

catSpec :: Spec
catSpec = do
  describe "rank-1" $ do
    it "stacks two tensors together" . property $ \i0 i1 -> do
      let
        t0 = constant dimsVec10 i0
        t1 = constant dimsVec10 i1
        t01 = (cat t0 t1 dim0)
      tensordata <$> t01 `shouldBe` Right (replicate 10 i0 ++ replicate 10 i1)

    it "stacks three tensors together" . property $ \i0 i1 i2 -> do
      let
        w2d = fromIntegral :: Int -> Double
        [d0, d1, d2] = w2d <$> [i0,i1,i2]

        t0 = constant dimsVec10 d0
        t1 = constant dimsVec10 d1
        t2 = constant dimsVec10 d2
        Right t12 = cat t1 t2 dim0

      tensordata <$> cat t0 t12 dim0 `shouldBe` Right (replicate 10 d0 ++ replicate 10 d1 ++ replicate 10 d2)
  where
    dimsVec10 = (dims :: Dims '[10])
    dim0 :: Word
    dim0 = dimVal (dim :: Dim 0)


