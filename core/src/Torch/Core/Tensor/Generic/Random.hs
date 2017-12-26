{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Generic.Random where

import qualified THFloatTensorRandom as T
import qualified THDoubleTensorRandom as T

import Torch.Core.Tensor.Generic.Internal

type SHOULD_BE_HASK_TYPE = CDouble

class GenericRandom t where
  random :: Ptr t -> Ptr CTHGenerator -> IO ()
  clampedRandom :: Ptr t -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()
  cappedRandom :: Ptr t -> Ptr CTHGenerator -> CLLong -> IO ()
  geometric :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> IO ()
  bernoulli :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> IO ()
  bernoulli_FloatTensor :: Ptr t -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()
  bernoulli_DoubleTensor :: Ptr t -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()
  uniform :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  normal :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  normal_means :: Ptr t -> Ptr CTHGenerator -> Ptr t -> SHOULD_BE_HASK_TYPE -> IO ()
  normal_stddevs :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> Ptr t -> IO ()
  normal_means_stddevs :: Ptr t -> Ptr CTHGenerator -> Ptr t -> Ptr t -> IO ()
  exponential :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> IO ()
  standard_gamma :: Ptr t -> Ptr CTHGenerator -> Ptr t -> IO ()
  cauchy :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  logNormal :: Ptr t -> Ptr CTHGenerator -> SHOULD_BE_HASK_TYPE -> SHOULD_BE_HASK_TYPE -> IO ()
  multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr t -> CInt -> CInt -> IO ()
  multinomialAliasSetup :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> IO ()
  multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr t -> IO ()

instance GenericRandom CTHDoubleTensor where
  random = T.c_THDoubleTensor_random
  clampedRandom = T.c_THDoubleTensor_clampedRandom
  cappedRandom = T.c_THDoubleTensor_cappedRandom
  geometric = T.c_THDoubleTensor_geometric
  bernoulli = T.c_THDoubleTensor_bernoulli
  bernoulli_FloatTensor = T.c_THDoubleTensor_bernoulli_FloatTensor
  bernoulli_DoubleTensor = T.c_THDoubleTensor_bernoulli_DoubleTensor
  uniform = T.c_THDoubleTensor_uniform
  normal = T.c_THDoubleTensor_normal
  normal_means = T.c_THDoubleTensor_normal_means
  normal_stddevs = T.c_THDoubleTensor_normal_stddevs
  normal_means_stddevs = T.c_THDoubleTensor_normal_means_stddevs
  exponential = T.c_THDoubleTensor_exponential
  standard_gamma = T.c_THDoubleTensor_standard_gamma
  cauchy = T.c_THDoubleTensor_cauchy
  logNormal = T.c_THDoubleTensor_logNormal
  multinomial = T.c_THDoubleTensor_multinomial
  multinomialAliasSetup = T.c_THDoubleTensor_multinomialAliasSetup
  multinomialAliasDraw = T.c_THDoubleTensor_multinomialAliasDraw

instance GenericRandom CTHFloatTensor where
  random = T.c_THFloatTensor_random
  clampedRandom = T.c_THFloatTensor_clampedRandom
  cappedRandom = T.c_THFloatTensor_cappedRandom
  geometric = T.c_THFloatTensor_geometric
  bernoulli = T.c_THFloatTensor_bernoulli
  bernoulli_FloatTensor = T.c_THFloatTensor_bernoulli_FloatTensor
  bernoulli_DoubleTensor = T.c_THFloatTensor_bernoulli_DoubleTensor
  uniform = T.c_THFloatTensor_uniform
  normal = T.c_THFloatTensor_normal
  normal_means = T.c_THFloatTensor_normal_means
  normal_stddevs = T.c_THFloatTensor_normal_stddevs
  normal_means_stddevs = T.c_THFloatTensor_normal_means_stddevs
  exponential = T.c_THFloatTensor_exponential
  standard_gamma = T.c_THFloatTensor_standard_gamma
  cauchy = T.c_THFloatTensor_cauchy
  logNormal = T.c_THFloatTensor_logNormal
  multinomial = T.c_THFloatTensor_multinomial
  multinomialAliasSetup = T.c_THFloatTensor_multinomialAliasSetup
  multinomialAliasDraw = T.c_THFloatTensor_multinomialAliasDraw

