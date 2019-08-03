{-# LANGUAGE ScopedTypeVariables #-}
module Torch.Core.Tensor.GenericSpec (spec) where

import Foreign (Ptr)
import Foreign.C.Types (CInt)

import THRandom (c_THGenerator_new)
import Torch.FFI.TH.Double.Tensor (c_THDoubleTensor_free)
import Torch.FFI.TH.Double.TensorMath (c_THDoubleTensor_stdall, c_THDoubleTensor_meanall, c_THDoubleTensor_maxall)
import Torch.FFI.TH.Double.TensorRandom (c_THDoubleTensor_normal, c_THDoubleTensor_uniform)

import Torch.Core.Tensor.Types (TensorDoubleRaw)
import Torch.Raw.Tensor.Generic

import Torch.Prelude.Extras


-- don't do this often, but since we are testing and there is a lot of code to port:
evilDim = unsafeSomeDims

main :: IO ()
main = hspec spec


spec :: Spec
spec = do
  describe "flatten" flattenSpec
  describe "constant'" constant'Spec
  describe "invlogit" invlogitSpec
  describe "randInit'" randInit'Spec


constant'Spec :: Spec
constant'Spec = do
  describe "filling constant tensors" $ do
    t :: TensorDoubleRaw <- runIO (constant' (evilDim [5]) 25)
    it "fills D1 tensors with the same values" $
      flatten t `shouldSatisfy` all (== 25)


invlogitSpec :: Spec
invlogitSpec = do
  describe "effects" $ do
    t :: TensorDoubleRaw <- runIO (constant' (evilDim [5, 3]) 25)
    t':: TensorDoubleRaw <- runIO (genericInvLogit t)
    it "inverts values up to an epsilon" $
      flatten t' `shouldSatisfy` all ((< 1e-10) . abs . (subtract 1))

    it "leaves the original tensor unchanged" $
      flatten t `shouldSatisfy` all (== 25)


randInit'Spec :: Spec
randInit'Spec = do
  describe "0 dimensional tensors" $ do
    gen <- runIO c_THGenerator_new
    rands :: [TensorDoubleRaw] <- runIO $ mapM (\_ -> randInit' gen (evilDim []) (-1.0) 3.0) [0..10]
    it "should only return empty tensors" $
      map flatten rands `shouldSatisfy` all null

  describe "1 dimensional tensors" $ do
    gen <- runIO c_THGenerator_new
    let runRandInit = randInit' gen (evilDim [5]) (-1.0) 3.0 :: IO TensorDoubleRaw
    rands0 <- runIO $ flatten <$> runRandInit
    rands1 <- runIO $ flatten <$> runRandInit
    rands2 <- runIO $ flatten <$> runRandInit
    it "should always return new values" $
      and (zipWith (/=) rands0 rands1)
      && and (zipWith (/=) rands1 rands2)

  describe "2 dimensional tensors" $ do
    assertSequencesAreUnique (evilDim [4, 5])

  describe "3 dimensional tensors" $ do
    assertSequencesAreUnique (evilDim [7,4,5])

  describe "4 dimensional tensors" $ do
    assertSequencesAreUnique (evilDim [0,7,4,5])
    assertSequencesAreUnique (evilDim [1,7,4,5])

 where
  assertSequencesAreUnique :: SomeDims -> Spec
  assertSequencesAreUnique d = do
    gen <- runIO c_THGenerator_new
    let runRandInit = randInit' gen d (-1.0) 3.0 :: IO TensorDoubleRaw
    rands' <- runIO $ mapM (const $ flatten <$> runRandInit) [0..10]
    let comp = zip (init rands') (tail rands')

    it "should always return new values" $
      and (and . uncurry (zipWith (/=)) <$> comp)


-- it would be nice to convert this into property checks
flattenSpec :: Spec
flattenSpec = do
  describe "0 dimensional tensors" $ do
    t <- runIO (mkTensor25 (evilDim []))
    it "returns the correct length"     $ length (flatten t) `shouldBe` 0
    it "returns the correct values"     $ flatten t   `shouldSatisfy` all (== 25)

  describe "1 dimensional tensors" $ do
    assertTmap (evilDim [5])

  describe "2 dimensional tensors" $ do
    assertTmap (evilDim [2,5])

  describe "3 dimensional tensors" $
    assertTmap (evilDim [4,2,5])

  describe "4 dimensional tensors" $
    assertTmap (evilDim [8,4,2,5])

 where
  mkTensor25 :: SomeDims -> IO TensorDoubleRaw
  mkTensor25 = flip constant' 25

  assertTmap :: SomeDims -> Spec
  assertTmap d = do
    t <- runIO (mkTensor25 d)
    it "returns the correct length"     $ length (flatten t) `shouldBe` fromIntegral (product' d)
    it "returns the correct values"     $ flatten t   `shouldSatisfy` all (== 25)
    where
      product' :: SomeDims -> Int
      product' (SomeDims d) = fromIntegral $ totalDim d



testsRawRandomScenario :: IO ()
testsRawRandomScenario = do
  gen <- c_THGenerator_new
  hspec $ do
    describe "random vectors" $ do
      it "uniform random is < bound" $ do
        t <- constant' (evilDim [1000]) 0.0
        c_Torch.FFI.TH.Double.Tensor_uniform t gen (-1.0) (1.0)
        c_Torch.FFI.TH.Double.Tensor_maxall t `shouldSatisfy` (< 1.001)
        c_Torch.FFI.TH.Double.Tensor_free t
      it "uniform random is > bound" $ do
        t <- constant' (evilDim [1000]) 0.0
        c_Torch.FFI.TH.Double.Tensor_uniform t gen (-1.0) (1.0)
        c_Torch.FFI.TH.Double.Tensor_maxall t `shouldSatisfy` (> (-1.001))
        c_Torch.FFI.TH.Double.Tensor_free t
      it "normal mean follows law of large numbers" $ do
        t <- constant' (evilDim [10000]) 0.0
        c_Torch.FFI.TH.Double.Tensor_normal t gen 1.55 0.25
        c_Torch.FFI.TH.Double.Tensor_meanall t `shouldSatisfy` (\x -> and [(x < 1.6), (x > 1.5)])
        c_Torch.FFI.TH.Double.Tensor_free t
      it "normal std follows law of large numbers" $ do
        t <- constant' (evilDim [10000]) 0.0
        c_Torch.FFI.TH.Double.Tensor_normal t gen 1.55 0.25
        c_Torch.FFI.TH.Double.Tensor_stdall t biased `shouldSatisfy` (\x -> and [(x < 0.3), (x > 0.2)])
        c_Torch.FFI.TH.Double.Tensor_free t
  where
    biased :: CInt
    biased = 0


