module TensorDoubleRandom (
  randomT
  , clampedRandomT
  , cappedRandomT
  , geometricT
  , bernoulliT
  , bernoulliFloatTensor
  , bernoulliDoubleTensor
  , uniformT
  , normalT
  , exponentialT
  , cauchyT
  , multinomialT
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

randomT :: TensorDouble_ -> RandGen -> IO TensorDouble_
randomT self gen = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_random s g
         )
    )
  pure self

clampedRandomT self gen minVal maxVal = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_clampedRandom s g minC maxC
         )
    )
  pure self
  where (minC, maxC) = (fromIntegral minVal, fromIntegral maxVal)

cappedRandomT :: TensorDouble_ -> RandGen -> Int -> IO TensorDouble_
cappedRandomT self gen maxVal = do
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
geometricT :: TensorDouble_ -> RandGen -> Double -> IO TensorDouble_
geometricT self gen p = do
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
bernoulliT :: TensorDouble_ -> RandGen -> Double -> IO TensorDouble_
bernoulliT self gen p = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_bernoulli s g pC
         )
    )
  pure self
  where pC = realToFrac p

bernoulliFloatTensor :: TensorDouble_ -> RandGen -> TensorFloat -> IO ()
bernoulliFloatTensor self gen p = do
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

bernoulliDoubleTensor :: TensorDouble_ -> RandGen -> TensorDouble_ -> IO TensorDouble_
bernoulliDoubleTensor self gen p = do
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

uniformT :: TensorDouble_ -> RandGen -> Double -> Double -> IO TensorDouble_
uniformT self gen a b = do
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

normalT :: TensorDouble_ -> RandGen -> Double -> Double -> IO TensorDouble_
normalT self gen mean stdv = do
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

exponentialT :: TensorDouble_ -> RandGen -> Double -> IO TensorDouble_
exponentialT self gen lambda = do
  withForeignPtr (tdTensor self)
    (\s ->
       withForeignPtr (rng gen)
         (\g ->
             c_THDoubleTensor_exponential s g lambdaC
         )
    )
  pure self
  where lambdaC = realToFrac lambda

cauchyT :: TensorDouble_ -> RandGen -> Double -> Double -> IO TensorDouble_
cauchyT self gen median sigma = do
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

logNormalT :: TensorDouble_ -> RandGen -> Double -> Double -> IO TensorDouble_
logNormalT self gen mean stdv = do
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


multinomialT :: TensorLong -> RandGen -> TensorDouble_ -> Int -> Bool -> IO TensorLong
multinomialT self gen prob_dist n_sample with_replacement = do
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
  let foo = tensorNew_ (D1 3)
  disp_ foo
  gen <- newRNG
  randomT foo gen
  disp_ foo
  pure ()
