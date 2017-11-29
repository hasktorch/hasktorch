{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Core.Tensor.Static.DoubleRandom
  ( tds_random
  , tds_clampedRandom
  , tds_cappedRandom
  , tds_geometric
  , tds_bernoulli
  , tds_bernoulliFloat
  , tds_bernoulliDouble
  , tds_uniform
  , tds_normal
  , tds_exponential
  , tds_cauchy
  , tds_multinomial

  , module Torch.Core.Random
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import Torch.Core.Tensor.Static.Double
-- import Torch.Core.Tensor.Double
import Torch.Core.Tensor.Dynamic.Long
import Torch.Core.Tensor.Raw
import Torch.Core.Tensor.Types
import Torch.Core.Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import THFloatTensor

import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits

-- |generate multivariate normal samples using Cholesky decomposition
tds_mvn :: (KnownNat r, KnownNat c) =>
  RandGen -> TDS '[c] -> TDS '[c,c] -> IO (TDS '[r,c])
tds_mvn rng mu cov = do
  let result = tds_new
  -- TODO: implement after core lapack functions implemented
  error "not implemented"
  pure result

-- TODO: get rid of self parameter arguments since they are overwritten

tds_random :: SingI d => RandGen -> IO (TDS d)
tds_random gen = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_random s g
         )
    )
  pure result

tds_clampedRandom gen minVal maxVal = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_clampedRandom s g minC maxC
         )
    )
  pure result
  where (minC, maxC) = (fromIntegral minVal, fromIntegral maxVal)

tds_cappedRandom :: SingI d => RandGen -> Int -> IO (TDS d)
tds_cappedRandom gen maxVal = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cappedRandom s g maxC
         )
    )
  pure result
  where maxC = fromIntegral maxVal

-- TH_API void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
tds_geometric :: SingI d => RandGen -> Double -> IO (TDS d)
tds_geometric gen p = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_geometric s g pC
         )
    )
  pure result
  where pC = realToFrac p

-- TH_API void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
tds_bernoulli :: SingI d => RandGen -> Double -> IO (TDS d)
tds_bernoulli gen p = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_bernoulli s g pC
         )
    )
  pure result
  where pC = realToFrac p

tds_bernoulliFloat :: SingI d => RandGen -> TensorFloat -> IO (TDS d)
tds_bernoulliFloat gen p = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_FloatTensor s g pTensor
              )
         )
    )
  pure result
  where pC = tfTensor p

tds_bernoulliDouble :: SingI d => RandGen -> TDS d -> IO (TDS d)
tds_bernoulliDouble gen p = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_DoubleTensor s g pTensor
              )
         )
    )
  pure result
  where pC = tdsTensor p

tds_uniform :: SingI d => RandGen -> Double -> Double -> IO (TDS d)
tds_uniform gen a b = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_uniform s g aC bC
         )
    )
  pure result
  where aC = realToFrac a
        bC = realToFrac b

tds_normal :: SingI d => RandGen -> Double -> Double -> IO (TDS d)
tds_normal gen mean stdv = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_normal s g meanC stdvC
         )
    )
  pure result
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

-- TH_API void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
-- TH_API void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
-- TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);

tds_exponential :: SingI d => RandGen -> Double -> IO (TDS d)
tds_exponential gen lambda = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_exponential s g lambdaC
         )
    )
  pure result
  where lambdaC = realToFrac lambda

tds_cauchy :: SingI d => RandGen -> Double -> Double -> IO (TDS d)
tds_cauchy gen median sigma = do
  let result = tds_new
  withForeignPtr (tdsTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cauchy s g medianC sigmaC
         )
    )
  pure result
  where medianC = realToFrac median
        sigmaC = realToFrac sigma

tds_logNormal :: SingI d => TDS d -> RandGen -> Double -> Double -> IO (TDS d)
tds_logNormal self gen mean stdv = do
  withForeignPtr (tdsTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_logNormal s g meanC stdvC
         )
    )
  pure self
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

tds_multinomial :: SingI d => RandGen -> TDS d -> Int -> Bool -> TensorDim Word -> IO (TensorLong)
tds_multinomial gen prob_dist n_sample with_replacement dim = do
  let result = tl_new dim
  withForeignPtr (tlTensor result)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (tdsTensor prob_dist)
              (\p ->
                 c_THDoubleTensor_multinomial s g p n_sampleC with_replacementC
              )
         )
    )
  pure result
  where n_sampleC = fromIntegral n_sample
        with_replacementC = if with_replacement then 1 else 0

-- TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
-- TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
-- #endif

