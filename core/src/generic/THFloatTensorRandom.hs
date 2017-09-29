{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorRandom (
    c_THFloatTensor_random,
    c_THFloatTensor_clampedRandom,
    c_THFloatTensor_cappedRandom,
    c_THFloatTensor_geometric,
    c_THFloatTensor_bernoulli,
    c_THFloatTensor_bernoulli_FloatTensor,
    c_THFloatTensor_bernoulli_DoubleTensor,
    c_THFloatTensor_uniform,
    c_THFloatTensor_normal,
    c_THFloatTensor_normal_means,
    c_THFloatTensor_normal_stddevs,
    c_THFloatTensor_normal_means_stddevs,
    c_THFloatTensor_exponential,
    c_THFloatTensor_cauchy,
    c_THFloatTensor_logNormal,
    c_THFloatTensor_multinomial,
    c_THFloatTensor_multinomialAliasSetup,
    c_THFloatTensor_multinomialAliasDraw) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensor_random : self _generator -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_random"
  c_THFloatTensor_random :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THFloatTensor_clampedRandom : self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_clampedRandom"
  c_THFloatTensor_clampedRandom :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THFloatTensor_cappedRandom : self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_cappedRandom"
  c_THFloatTensor_cappedRandom :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THFloatTensor_geometric : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_geometric"
  c_THFloatTensor_geometric :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_bernoulli : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_bernoulli"
  c_THFloatTensor_bernoulli :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_bernoulli_FloatTensor"
  c_THFloatTensor_bernoulli_FloatTensor :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THFloatTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_bernoulli_DoubleTensor"
  c_THFloatTensor_bernoulli_DoubleTensor :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THFloatTensor_uniform : self _generator a b -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_uniform"
  c_THFloatTensor_uniform :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_normal : self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_normal"
  c_THFloatTensor_normal :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_normal_means : self gen means stddev -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_normal_means"
  c_THFloatTensor_normal_means :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_normal_stddevs"
  c_THFloatTensor_normal_stddevs :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_normal_means_stddevs"
  c_THFloatTensor_normal_means_stddevs :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_exponential : self _generator lambda -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_exponential"
  c_THFloatTensor_exponential :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_cauchy : self _generator median sigma -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_cauchy"
  c_THFloatTensor_cauchy :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_logNormal : self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_logNormal"
  c_THFloatTensor_logNormal :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_multinomial"
  c_THFloatTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_multinomialAliasSetup"
  c_THFloatTensor_multinomialAliasSetup :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall unsafe "THTensorRandom.h THFloatTensor_multinomialAliasDraw"
  c_THFloatTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()