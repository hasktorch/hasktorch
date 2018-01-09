{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Raw.Tensor.Random
  ( THTensorRandom(..)
  , module X
  ) where

import qualified THFloatTensorRandom as T
import qualified THDoubleTensorRandom as T

import Torch.Raw.Internal as X

class THTensorRandom t where
  c_random :: Ptr t -> Ptr CTHGenerator -> IO ()
  c_clampedRandom :: Ptr t -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()
  c_cappedRandom :: Ptr t -> Ptr CTHGenerator -> CLLong -> IO ()
  c_geometric :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> IO ()
  c_bernoulli :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> IO ()
  c_bernoulli_FloatTensor :: Ptr t -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()
  c_bernoulli_DoubleTensor :: Ptr t -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()
  c_uniform :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_normal :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_normal_means :: Ptr t -> Ptr CTHGenerator -> Ptr t -> HaskAccReal t -> IO ()
  c_normal_stddevs :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> Ptr t -> IO ()
  c_normal_means_stddevs :: Ptr t -> Ptr CTHGenerator -> Ptr t -> Ptr t -> IO ()
  c_exponential :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> IO ()
  -- c_standard_gamma :: Ptr t -> Ptr CTHGenerator -> Ptr t -> IO ()
  c_cauchy :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_logNormal :: Ptr t -> Ptr CTHGenerator -> HaskAccReal t -> HaskAccReal t -> IO ()
  c_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr t -> CInt -> CInt -> IO ()
  c_multinomialAliasSetup :: Ptr t -> Ptr CTHLongTensor -> Ptr t -> IO ()
  c_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr t -> IO ()

instance THTensorRandom CTHDoubleTensor where
  c_random = T.c_THDoubleTensor_random
  c_clampedRandom = T.c_THDoubleTensor_clampedRandom
  c_cappedRandom = T.c_THDoubleTensor_cappedRandom
  c_geometric = T.c_THDoubleTensor_geometric
  c_bernoulli = T.c_THDoubleTensor_bernoulli
  c_bernoulli_FloatTensor = T.c_THDoubleTensor_bernoulli_FloatTensor
  c_bernoulli_DoubleTensor = T.c_THDoubleTensor_bernoulli_DoubleTensor
  c_uniform = T.c_THDoubleTensor_uniform
  c_normal = T.c_THDoubleTensor_normal
  c_normal_means = T.c_THDoubleTensor_normal_means
  c_normal_stddevs = T.c_THDoubleTensor_normal_stddevs
  c_normal_means_stddevs = T.c_THDoubleTensor_normal_means_stddevs
  c_exponential = T.c_THDoubleTensor_exponential
  -- c_standard_gamma = T.c_THDoubleTensor_standard_gamma
  c_cauchy = T.c_THDoubleTensor_cauchy
  c_logNormal = T.c_THDoubleTensor_logNormal
  c_multinomial = T.c_THDoubleTensor_multinomial
  c_multinomialAliasSetup = T.c_THDoubleTensor_multinomialAliasSetup
  c_multinomialAliasDraw = T.c_THDoubleTensor_multinomialAliasDraw

instance THTensorRandom CTHFloatTensor where
  c_random = T.c_THFloatTensor_random
  c_clampedRandom = T.c_THFloatTensor_clampedRandom
  c_cappedRandom = T.c_THFloatTensor_cappedRandom
  c_geometric = T.c_THFloatTensor_geometric
  c_bernoulli = T.c_THFloatTensor_bernoulli
  c_bernoulli_FloatTensor = T.c_THFloatTensor_bernoulli_FloatTensor
  c_bernoulli_DoubleTensor = T.c_THFloatTensor_bernoulli_DoubleTensor
  c_uniform = T.c_THFloatTensor_uniform
  c_normal = T.c_THFloatTensor_normal
  c_normal_means = T.c_THFloatTensor_normal_means
  c_normal_stddevs = T.c_THFloatTensor_normal_stddevs
  c_normal_means_stddevs = T.c_THFloatTensor_normal_means_stddevs
  c_exponential = T.c_THFloatTensor_exponential
  -- c_standard_gamma = T.c_THFloatTensor_standard_gamma
  c_cauchy = T.c_THFloatTensor_cauchy
  c_logNormal = T.c_THFloatTensor_logNormal
  c_multinomial = T.c_THFloatTensor_multinomial
  c_multinomialAliasSetup = T.c_THFloatTensor_multinomialAliasSetup
  c_multinomialAliasDraw = T.c_THFloatTensor_multinomialAliasDraw

