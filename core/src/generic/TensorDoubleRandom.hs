module TensorDoubleRandom (
  td_random
  , td_clampedRandom
  , td_cappedRandom
  , td_geometric
  , td_bernoulli
  , td_bernoulliFloat
  , td_bernoulliDouble
  , td_uniform
  , td_normal
  , td_exponential
  , td_cauchy
  , td_multinomial
  ) where

import Foreign
import Foreign.C.Types
import Foreign.Ptr
import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import GHC.Ptr (FunPtr)
import System.IO.Unsafe (unsafePerformIO)

import TensorDouble
import TensorRaw
import TensorTypes
import TensorUtils
import Random

import THTypes
import THRandom
import THDoubleTensor
import THDoubleTensorMath
import THDoubleTensorRandom

import THFloatTensor

td_random :: TensorDouble -> RandGen -> IO TensorDouble
td_random self gen = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_random s g
         )
    )
  pure self

td_clampedRandom self gen minVal maxVal = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_clampedRandom s g minC maxC
         )
    )
  pure self
  where (minC, maxC) = (fromIntegral minVal, fromIntegral maxVal)

td_cappedRandom :: TensorDouble -> RandGen -> Int -> IO TensorDouble
td_cappedRandom self gen maxVal = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cappedRandom s g maxC
         )
    )
  pure self
  where maxC = fromIntegral maxVal

-- TH_API void THTensor_(geometric)(THTensor *self, THGenerator *_generator, double p);
td_geometric :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_geometric self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_geometric s g pC
         )
    )
  pure self
  where pC = realToFrac p

-- TH_API void THTensor_(bernoulli)(THTensor *self, THGenerator *_generator, double p);
td_bernoulli :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_bernoulli self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_bernoulli s g pC
         )
    )
  pure self
  where pC = realToFrac p

td_bernoulliFloat :: TensorDouble -> RandGen -> TensorFloat -> IO ()
td_bernoulliFloat self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_FloatTensor s g pTensor
              )
         )
    )
  where pC = tfTensor p

td_bernoulliDouble :: TensorDouble -> RandGen -> TensorDouble -> IO TensorDouble
td_bernoulliDouble self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (pC)
              (\pTensor ->
                 c_THDoubleTensor_bernoulli_DoubleTensor s g pTensor
              )
         )
    )
  pure self
  where pC = tdTensor p

td_uniform :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_uniform self gen a b = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_uniform s g aC bC
         )
    )
  pure self
  where aC = realToFrac a
        bC = realToFrac b

td_normal :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_normal self gen mean stdv = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_normal s g meanC stdvC
         )
    )
  pure self
  where meanC = realToFrac mean
        stdvC = realToFrac stdv

-- TH_API void THTensor_(normal_means)(THTensor *self, THGenerator *gen, THTensor *means, double stddev);
-- TH_API void THTensor_(normal_stddevs)(THTensor *self, THGenerator *gen, double mean, THTensor *stddevs);
-- TH_API void THTensor_(normal_means_stddevs)(THTensor *self, THGenerator *gen, THTensor *means, THTensor *stddevs);

td_exponential :: TensorDouble -> RandGen -> Double -> IO TensorDouble
td_exponential self gen lambda = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_exponential s g lambdaC
         )
    )
  pure self
  where lambdaC = realToFrac lambda

td_cauchy :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_cauchy self gen median sigma = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_cauchy s g medianC sigmaC
         )
    )
  pure self
  where medianC = realToFrac median
        sigmaC = realToFrac sigma

td_logNormal :: TensorDouble -> RandGen -> Double -> Double -> IO TensorDouble
td_logNormal self gen mean stdv = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_logNormal s g meanC stdvC
         )
    )
  pure self
  where meanC = realToFrac mean
        stdvC = realToFrac stdv


td_multinomial :: TensorLong -> RandGen -> TensorDouble -> Int -> Bool -> IO TensorLong
td_multinomial self gen prob_dist n_sample with_replacement = do
  withForeignPtr (tlTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
            withForeignPtr (tdTensor prob_dist)
              (\p ->
                 c_THDoubleTensor_multinomial s g p n_sampleC with_replacementC
              )
         )
    )
  pure self
  where n_sampleC = fromIntegral n_sample
        with_replacementC = if with_replacement then 1 else 0

-- TH_API void THTensor_(multinomialAliasSetup)(THTensor *prob_dist, THLongTensor *J, THTensor *q);
-- TH_API void THTensor_(multinomialAliasDraw)(THLongTensor *self, THGenerator *_generator, THLongTensor *J, THTensor *q);
-- #endif

test = do
  let t = tdNew (D1 3)
  disp t
  gen <- newRNG
  td_random t gen
  disp t
  pure ()
