{-# LANGUAGE ForeignFunctionInterface#-}

module THByteTensorRandom (
    c_THByteTensor_random,
    c_THByteTensor_clampedRandom,
    c_THByteTensor_cappedRandom,
    c_THByteTensor_geometric,
    c_THByteTensor_bernoulli,
    c_THByteTensor_bernoulli_FloatTensor,
    c_THByteTensor_bernoulli_DoubleTensor,
    c_THByteTensor_uniform,
    c_THByteTensor_normal,
    c_THByteTensor_normal_means,
    c_THByteTensor_normal_stddevs,
    c_THByteTensor_normal_means_stddevs,
    c_THByteTensor_exponential,
    c_THByteTensor_cauchy,
    c_THByteTensor_logNormal,
    c_THByteTensor_multinomial,
    c_THByteTensor_multinomialAliasSetup,
    c_THByteTensor_multinomialAliasDraw,
    c_THByteTensor_getRNGState,
    c_THByteTensor_setRNGState) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THByteTensor_random"
  c_THByteTensor_random :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THByteTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THByteTensor_clampedRandom"
  c_THByteTensor_clampedRandom :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THByteTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THByteTensor_cappedRandom"
  c_THByteTensor_cappedRandom :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THByteTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_geometric"
  c_THByteTensor_geometric :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THByteTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli"
  c_THByteTensor_bernoulli :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THByteTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_FloatTensor"
  c_THByteTensor_bernoulli_FloatTensor :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THByteTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_DoubleTensor"
  c_THByteTensor_bernoulli_DoubleTensor :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THByteTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THByteTensor_uniform"
  c_THByteTensor_uniform :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THByteTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THByteTensor_normal"
  c_THByteTensor_normal :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THByteTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THByteTensor_normal_means"
  c_THByteTensor_normal_means :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> (Ptr CTHByteTensor) -> CDouble -> IO ()

-- |c_THByteTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THByteTensor_normal_stddevs"
  c_THByteTensor_normal_stddevs :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THByteTensor_normal_means_stddevs"
  c_THByteTensor_normal_means_stddevs :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> (Ptr CTHByteTensor) -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THByteTensor_exponential"
  c_THByteTensor_exponential :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THByteTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THByteTensor_cauchy"
  c_THByteTensor_cauchy :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THByteTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THByteTensor_logNormal"
  c_THByteTensor_logNormal :: (Ptr CTHByteTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THByteTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THByteTensor_multinomial"
  c_THByteTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHByteTensor) -> CInt -> CInt -> IO ()

-- |c_THByteTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THByteTensor_multinomialAliasSetup"
  c_THByteTensor_multinomialAliasSetup :: (Ptr CTHByteTensor) -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THByteTensor_multinomialAliasDraw"
  c_THByteTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_getRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_getRNGState"
  c_THByteTensor_getRNGState :: Ptr CTHGenerator -> (Ptr CTHByteTensor) -> IO ()

-- |c_THByteTensor_setRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_setRNGState"
  c_THByteTensor_setRNGState :: Ptr CTHGenerator -> (Ptr CTHByteTensor) -> IO ()