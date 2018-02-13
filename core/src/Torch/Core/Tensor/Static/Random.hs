{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}
module Torch.Core.Tensor.Static.Random
  ( random
  , clampedRandom
  , cappedRandom
  , geometric
  , bernoulli
  , bernoulli_FloatTensor
  , bernoulli_DoubleTensor
  ) where

import Torch.Class.C.Internal (AsDynamic)
import GHC.Int (Int64)
import THTypes (CTHGenerator)
import Foreign (Ptr)

import Torch.Core.Tensor.Static
import Torch.Core.Tensor.Dim
import Torch.Core.Tensor.Dynamic.Random (TensorRandom)
import qualified Torch.Core.Tensor.Dynamic as Dynamic

-- FIXME: (stites) - I think we can replace all of these with the derived dynamic instance and just implement the bernoulli_Double-/Float-Tensors.
type RandomConstraint t d = (StaticConstraint (t d), TensorRandom (AsDynamic (t d)), Dimensions d)

random :: RandomConstraint t d => Ptr CTHGenerator -> IO (t d)
random g = withInplace (`Dynamic.random` g)

clampedRandom :: RandomConstraint t d => Ptr CTHGenerator -> Int64 -> Int64 -> IO (t d)
clampedRandom g a b = withInplace $ \res -> Dynamic.clampedRandom res g a b

cappedRandom :: RandomConstraint t d => Ptr CTHGenerator -> Int64 -> IO (t d)
cappedRandom g a = withInplace $ \res -> Dynamic.cappedRandom res g a

geometric :: RandomConstraint t d => Ptr CTHGenerator -> Double -> IO (t d)
geometric g a = withInplace $ \res -> Dynamic.geometric res g a

bernoulli :: RandomConstraint t d => Ptr CTHGenerator -> Double -> IO (t d)
bernoulli g a = withInplace $ \res -> Dynamic.bernoulli res g a

-- (stites): I think these functions take a distribution as input, but I'm not sure how the dimensions need to line up.
-- TODO: use static tensors / singletons to encode distribution information.
bernoulli_FloatTensor :: RandomConstraint t d => Ptr CTHGenerator -> Dynamic.FloatTensor -> IO (t d)
bernoulli_FloatTensor g a = withInplace $ \res -> Dynamic.bernoulli_FloatTensor res g a

bernoulli_DoubleTensor :: RandomConstraint t d => Ptr CTHGenerator -> Dynamic.DoubleTensor -> IO (t d)
bernoulli_DoubleTensor g a = withInplace $ \res -> Dynamic.bernoulli_DoubleTensor res g a

{-
-- | generate correlated multivariate normal samples by specifying eigendecomposition
tds_mvn :: forall n p . (KnownNatDim n, KnownNatDim p) =>
  RandGen -> TDS '[p] -> TDS '[p,p] -> TDS '[p] -> IO (TDS '[n, p])
tds_mvn gen mu eigenvectors eigenvalues = do
  let offset = tds_expand mu :: TDS '[n, p]
  samps <- tds_normal gen 0.0 1.0 :: IO (TDS '[p, n])
  let result = tds_trans ((tds_trans eigenvectors)
                          !*! (tds_diag eigenvalues)
                          !*! eigenvectors
                          !*! samps) + offset
  pure result

test_mvn :: IO ()
test_mvn = do
  gen <- newRNG
  let eigenvectors = tds_fromList [1, 1, 1, 1, 1, 1, 0, 0, 0] :: TDS '[3,3]
  tds_p eigenvectors
  let eigenvalues = tds_fromList [1, 1, 1] :: TDS '[3]
  tds_p eigenvalues
  let mu = tds_fromList [0.0, 0.0, 0.0] :: TDS '[3]
  result <- tds_mvn gen mu eigenvectors eigenvalues :: IO (TDS '[10, 3])
  tds_p result


tds_normal :: SingDimensions d => RandGen -> Double -> Double -> IO (TDS d)
tds_normal gen mean stdv = do
  let result = tds_new
  runManaged $ do
    s <- managed (withForeignPtr (tdsTensor result))
    g <- managed (withForeignPtr (rng gen))
    liftIO (c_THDoubleTensor_normal s g meanC stdvC)
  pure result
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

-- TH_API void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
-- TH_API void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
-- TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);

tds_exponential :: SingDimensions d => RandGen -> Double -> IO (TDS d)
tds_exponential gen lambda = do
  let result = tds_new
  runManaged $ do
    s <- managed (withForeignPtr (tdsTensor result))
    g <- managed (withForeignPtr (rng gen))
    liftIO (c_THDoubleTensor_exponential s g lambdaC)
  pure result
  where lambdaC = realToFrac lambda

tds_cauchy :: SingDimensions d => RandGen -> Double -> Double -> IO (TDS d)
tds_cauchy gen median sigma = do
  let result = tds_new
  runManaged $ do
    s <- managed (withForeignPtr (tdsTensor result))
    g <- managed (withForeignPtr (rng gen))
    liftIO (c_THDoubleTensor_cauchy s g medianC sigmaC)
  pure result
  where medianC = realToFrac median
        sigmaC = realToFrac sigma

tds_logNormal :: SingDimensions d => RandGen -> Double -> Double -> IO (TDS d)
tds_logNormal gen mean stdv = do
  let result = tds_new
  runManaged $ do
    s <- managed (withForeignPtr (tdsTensor result))
    g <- managed (withForeignPtr (rng gen))
    liftIO (c_THDoubleTensor_logNormal s g meanC stdvC)
  pure result
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

tds_multinomial :: SingDimensions d => RandGen -> TDS d -> Int -> Bool -> SomeDims -> IO TensorLong
tds_multinomial gen prob_dist n_sample with_replacement dim = do
  let result = tl_new dim
  runManaged $ do
    s <- managed (withForeignPtr (tlTensor result))
    g <- managed (withForeignPtr (rng gen))
    p <- managed (withForeignPtr (tdsTensor prob_dist))
    liftIO (c_THDoubleTensor_multinomial s g p n_sampleC with_replacementC)
  pure result
  where n_sampleC = fromIntegral n_sample
        with_replacementC = if with_replacement then 1 else 0

-- TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
-- TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
-- #endif
-}
