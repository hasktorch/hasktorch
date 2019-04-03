{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ExistentialQuantification #-}
module Torch.Indef.Static.Tensor.MathSpec where

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math

import Numeric.Dimensions
import Test.Hspec
import Test.QuickCheck
import Data.List.NonEmpty (NonEmpty(..))
import qualified Data.List.NonEmpty as NE
import qualified Data.Singletons.Prelude.List as Sing hiding (All, type (++))

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
  describe "zerosLike" zerosLikeSpec
  describe "onesLike" onesLikeSpec
  describe "cat" catSpec
  describe "catArray0" catArray0Spec

fill_Spec :: Spec
fill_Spec = do
  it "fills inplace" $ property $ \i -> do
    let ten = new :: Tensor '[2, 3]
    fill_ ten i
    tensordata ten `shouldSatisfy` (all (== i))

zero_Spec :: Spec
zero_Spec = do
  it "fills zero inplace" $ do
    let ten = new :: Tensor '[2, 3]
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
  it "returns the number of elements in the tensor" $ do
    numel (new :: Tensor '[2,3]) `shouldBe` 6
    numel (new :: Tensor '[3,4]) `shouldBe` 12
    numel (new :: Tensor '[323,401]) `shouldBe` (323*401)

zerosLikeSpec :: Spec
zerosLikeSpec =
  it "forms a new tensor with the same shape as the argument tensor and fills it with zeros" $ do
    let t = zerosLike :: Tensor '[2, 3]
    tensordata t `shouldSatisfy` all (== 0)

onesLikeSpec :: Spec
onesLikeSpec =
  it "forms a new tensor with the same shape as the argument tensor and fills it with ones" $ do
    let t = onesLike :: Tensor '[2, 3]
    tensordata t `shouldSatisfy` all (== 1)

catSpec :: Spec
catSpec = do
  describe "rank-1" $ do
    it "stacks two tensors together" . property $ \i0 i1 -> do
      let
        t0 = constant i0 :: Tensor '[10]
        t1 = constant i1 :: Tensor '[10]
      tensordata (cat t0 t1 dim0) `shouldBe` (replicate 10 i0 ++ replicate 10 i1)

    it "stacks three tensors together" . property $ \i0 i1 i2 -> do
      let
        w2d = fromIntegral :: Int -> Double
        [d0, d1, d2] = w2d <$> [i0,i1,i2]

        t0 = constant d0 :: Tensor '[10]
        t1 = constant d1 :: Tensor '[10]
        t2 = constant d2 :: Tensor '[10]
        t12 = cat t1 t2 dim0

      tensordata (cat t0 t12 dim0) `shouldBe` (replicate 10 d0 ++ replicate 10 d1 ++ replicate 10 d2)
  where
    dim0 = (dim :: Dim 0)

catArray0Spec :: Spec
catArray0Spec = do
  describe "rank-1" $ do
    it "stacks two tensors together" . property $ \i0 i1 -> do
      let
        t0 = constant i0 :: Tensor '[10]
        t1 = constant i1 :: Tensor '[10]
      tensordata <$> (catArray0 (t0:|[t1]) :: Either String (Tensor '[20])) `shouldBe` Right (replicate 10 i0 ++ replicate 10 i1)

    it "stacks three tensors together" . property $ \i0 i1 i2 -> do
      let
        w2d = fromIntegral :: Int -> Double
        [d0, d1, d2] = w2d <$> [i0,i1,i2]

        t0 = constant d0 :: Tensor '[10]
        t1 = constant d1 :: Tensor '[10]
        t2 = constant d2 :: Tensor '[10]

      tensordata <$> (catArray0 (t0:|[t1, t2]) :: Either String (Tensor '[30])) `shouldBe` Right (replicate 10 d0 ++ replicate 10 d1 ++ replicate 10 d2)
  where
    dim0 = (dim :: Dim 0)





{-
-
-- NOTE: In C, if the dimension is not specified or if it is -1, it is the maximum
-- last dimension over all input tensors, except if all tensors are empty, then it is 1.
catArray :: (Dimensions d) => [Dynamic] -> Word -> IO (Tensor d)
catArray ts dv = let r = empty in Dynamic._catArray (asDynamic r) ts dv >> pure r

catArray0 :: (Dimensions d, Dimensions d2) => [Tensor d2] -> Tensor d
catArray0 ts = unsafeDupablePerformIO $ catArray (asDynamic <$> ts) 0
{-# NOINLINE catArray0 #-}

{-
catArray_
  :: forall d ls rs out n
  .  All Dimensions '[out]
  => out ~ (rs ++ '[Length '[Tensor d]] ++ ls)
  => '(ls, rs) ~ Sing.SplitAt n d

  => Sing.SList '[Tensor d]
  -> Dim n
  -> IO (Tensor out)
catArray_ ts dv
  = -- fmap asStatic
    catArray
    (asDynamic <$> (fromSing ts :: [Tensor d]))
    (fromIntegral $ dimVal dv)

-- data Sing (z :: [a]) where
--     SNil :: Sing ([] :: [k])
--     SCons :: Sing (n ': n)

singToList :: forall k ks k2 x . Sing.SList '[x] -> [x]
singToList sl = go [] sl
 where
  go :: [x] -> Sing.SList '[x] -> [x]
  -- go acc Sing.SNil = acc
  -- go acc (Sing.SNil :: Sing.SList ('[] :: [x])) = acc
  -- go acc (Sing.SConst :: Sing.Sing '[]) = acc
  go acc (Sing.SCons k ks) = go acc ks

    -- | fromSing (Sing.sNull sl) = reverse acc
    -- | otherwise = go (fromSing (Sing.sHead sl):acc) (Sing.sTail acc)
  -- Sing.SNil = reverse acc
  -- go acc (Sing.SCons sval rest) = go (fromSing sval:acc) rest
-}

-}

