{-# LANGUAGE ForeignFunctionInterface#-}

module THHalfTensorRandom (
    c_THHalfTensor_random,
    c_THHalfTensor_clampedRandom,
    c_THHalfTensor_cappedRandom,
    c_THHalfTensor_geometric,
    c_THHalfTensor_bernoulli,
    c_THHalfTensor_bernoulli_FloatTensor,
    c_THHalfTensor_bernoulli_DoubleTensor,
    c_THHalfTensor_uniform,
    c_THHalfTensor_normal,
    c_THHalfTensor_normal_means,
    c_THHalfTensor_normal_stddevs,
    c_THHalfTensor_normal_means_stddevs,
    c_THHalfTensor_exponential,
    c_THHalfTensor_cauchy,
    c_THHalfTensor_logNormal,
    c_THHalfTensor_multinomial,
    c_THHalfTensor_multinomialAliasSetup,
    c_THHalfTensor_multinomialAliasDraw,
    c_THHalfTensor_getRNGState,
    c_THHalfTensor_setRNGState) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THHalfTensor_random"
  c_THHalfTensor_random :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THHalfTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_clampedRandom"
  c_THHalfTensor_clampedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_cappedRandom"
  c_THHalfTensor_cappedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THHalfTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_geometric"
  c_THHalfTensor_geometric :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli"
  c_THHalfTensor_bernoulli :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_FloatTensor"
  c_THHalfTensor_bernoulli_FloatTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THHalfTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_DoubleTensor"
  c_THHalfTensor_bernoulli_DoubleTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THHalfTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THHalfTensor_uniform"
  c_THHalfTensor_uniform :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THHalfTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THHalfTensor_normal"
  c_THHalfTensor_normal :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THHalfTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THHalfTensor_normal_means"
  c_THHalfTensor_normal_means :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> CDouble -> IO ()

-- |c_THHalfTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THHalfTensor_normal_stddevs"
  c_THHalfTensor_normal_stddevs :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THHalfTensor_normal_means_stddevs"
  c_THHalfTensor_normal_means_stddevs :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THHalfTensor_exponential"
  c_THHalfTensor_exponential :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THHalfTensor_cauchy"
  c_THHalfTensor_cauchy :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THHalfTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THHalfTensor_logNormal"
  c_THHalfTensor_logNormal :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THHalfTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THHalfTensor_multinomial"
  c_THHalfTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> CInt -> CInt -> IO ()

-- |c_THHalfTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THHalfTensor_multinomialAliasSetup"
  c_THHalfTensor_multinomialAliasSetup :: (Ptr CTHHalfTensor) -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THHalfTensor_multinomialAliasDraw"
  c_THHalfTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_getRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THHalfTensor_getRNGState"
  c_THHalfTensor_getRNGState :: Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_setRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THHalfTensor_setRNGState"
  c_THHalfTensor_setRNGState :: Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> IO ()