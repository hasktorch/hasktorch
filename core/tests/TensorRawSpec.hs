{-# LANGUAGE ScopedTypeVariables #-}
module TensorRawSpec (spec) where

import Foreign (Ptr)
import Test.Hspec
import Test.QuickCheck

import TensorRaw
import TensorTypes (TensorDim(..))
import Torch.Core.Tensor.Raw (toList)
import qualified THRandom as R (c_THGenerator_new)

main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "tensorRaw" tensorRawSpec
  describe "invlogit" invlogitSpec
  describe "randInitRaw" randInitRawSpec


tensorRawSpec :: Spec
tensorRawSpec = do
  describe "filling constant tensors" $ do
    t <- runIO (tensorRaw (D1 5) 25)
    it "fills D1 tensors with the same values" $
      toList t `shouldSatisfy` all (== 25)


invlogitSpec :: Spec
invlogitSpec = do
  describe "effects" $ do
    t  <- runIO (tensorRaw (D2 (5, 3)) 25)
    t' <- runIO (invlogit t)
    it "inverts values up to an epsilon" $
      toList t' `shouldSatisfy` all ((< 1e-10) . abs . (subtract 1))

    it "leaves the original tensor unchanged" $
      toList t `shouldSatisfy` all (== 25)


randInitRawSpec :: Spec
randInitRawSpec = do
  describe "0 dimensional tensors" $ do
    gen <- runIO R.c_THGenerator_new
    rands <- runIO $ mapM (\_ -> randInitRaw gen D0 (-1.0) 3.0) [0..10]
    it "should only return empty tensors" $
      map toList rands `shouldSatisfy` all null

  describe "1 dimensional tensors" $ do
    gen <- runIO R.c_THGenerator_new
    let runRandInit = randInitRaw gen (D1 5) (-1.0) 3.0
    rands0 <- runIO $ toList <$> runRandInit
    rands1 <- runIO $ toList <$> runRandInit
    rands2 <- runIO $ toList <$> runRandInit
    it "should always return new values" $
      and (zipWith (/=) rands0 rands1)
      && and (zipWith (/=) rands1 rands2)

  describe "2 dimensional tensors" $ do
    assertSequencesAreUnique (D2 (4,5))

  describe "3 dimensional tensors" $ do
    assertSequencesAreUnique (D3 (7,4,5))

  describe "4 dimensional tensors" $ do
    assertSequencesAreUnique (D4 (0,7,4,5))
    assertSequencesAreUnique (D4 (1,7,4,5))

 where
  assertSequencesAreUnique :: TensorDim Word -> Spec
  assertSequencesAreUnique d = do
    gen <- runIO R.c_THGenerator_new
    let runRandInit = randInitRaw gen d (-1.0) 3.0
    rands' <- runIO $ mapM (const $ toList <$> runRandInit) [0..10]
    let comp = zip (init rands') (tail rands')

    it "should always return new values" $
      and (and . uncurry (zipWith (/=)) <$> comp)

