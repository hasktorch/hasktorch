module Torch.Core.Tensor.RawSpec (spec) where

import Foreign (Ptr)
import Test.Hspec
import Test.QuickCheck

import Torch.Core.Tensor.Types (TensorDim(..), TensorDoubleRaw)
import Torch.Core.Tensor.Raw

import qualified THRandom as R (c_THGenerator_new)


main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "toList" toListSpec
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


-- it would be nice to convert this into property checks
toListSpec :: Spec
toListSpec = do
  describe "0 dimensional tensors" $ do
    t <- runIO (mkTensor25 D0)
    it "returns the correct length"     $ length (toList t) `shouldBe` 0
    it "returns the correct values"     $ toList t   `shouldSatisfy` all (== 25)

  describe "1 dimensional tensors" $ do
    assertTmap (D1 5)

  describe "2 dimensional tensors" $ do
    assertTmap (D2 (2,5))

  describe "3 dimensional tensors" $
    assertTmap (D3 (4,2,5))

  describe "4 dimensional tensors" $
    assertTmap (D4 (8,4,2,5))

 where
  mkTensor25 :: TensorDim Word -> IO TensorDoubleRaw
  mkTensor25 = flip tensorRaw 25

  assertTmap :: TensorDim Word -> Spec
  assertTmap d = do
    t <- runIO (mkTensor25 d)
    it "returns the correct length"     $ length (toList t) `shouldBe` fromIntegral (product d)
    it "returns the correct values"     $ toList t   `shouldSatisfy` all (== 25)


