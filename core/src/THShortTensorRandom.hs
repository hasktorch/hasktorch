{-# LANGUAGE ForeignFunctionInterface#-}

module THShortTensorRandom (
    c_THShortTensor_random,
    c_THShortTensor_clampedRandom,
    c_THShortTensor_cappedRandom,
    c_THShortTensor_geometric,
    c_THShortTensor_bernoulli,
    c_THShortTensor_bernoulli_FloatTensor,
    c_THShortTensor_bernoulli_DoubleTensor,
    c_THShortTensor_uniform,
    c_THShortTensor_normal,
    c_THShortTensor_normal_means,
    c_THShortTensor_normal_stddevs,
    c_THShortTensor_normal_means_stddevs,
    c_THShortTensor_exponential,
    c_THShortTensor_cauchy,
    c_THShortTensor_logNormal,
    c_THShortTensor_multinomial,
    c_THShortTensor_multinomialAliasSetup,
    c_THShortTensor_multinomialAliasDraw,
    c_THShortTensor_getRNGState,
    c_THShortTensor_setRNGState) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THShortTensor_random"
  c_THShortTensor_random :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THShortTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THShortTensor_clampedRandom"
  c_THShortTensor_clampedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THShortTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THShortTensor_cappedRandom"
  c_THShortTensor_cappedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THShortTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_geometric"
  c_THShortTensor_geometric :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli"
  c_THShortTensor_bernoulli :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_FloatTensor"
  c_THShortTensor_bernoulli_FloatTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THShortTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_DoubleTensor"
  c_THShortTensor_bernoulli_DoubleTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THShortTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THShortTensor_uniform"
  c_THShortTensor_uniform :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THShortTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THShortTensor_normal"
  c_THShortTensor_normal :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THShortTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THShortTensor_normal_means"
  c_THShortTensor_normal_means :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> (Ptr CTHShortTensor) -> CDouble -> IO ()

-- |c_THShortTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THShortTensor_normal_stddevs"
  c_THShortTensor_normal_stddevs :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THShortTensor_normal_means_stddevs"
  c_THShortTensor_normal_means_stddevs :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THShortTensor_exponential"
  c_THShortTensor_exponential :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THShortTensor_cauchy"
  c_THShortTensor_cauchy :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THShortTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THShortTensor_logNormal"
  c_THShortTensor_logNormal :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THShortTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THShortTensor_multinomial"
  c_THShortTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHShortTensor) -> CInt -> CInt -> IO ()

-- |c_THShortTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THShortTensor_multinomialAliasSetup"
  c_THShortTensor_multinomialAliasSetup :: (Ptr CTHShortTensor) -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THShortTensor_multinomialAliasDraw"
  c_THShortTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_getRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THShortTensor_getRNGState"
  c_THShortTensor_getRNGState :: Ptr CTHGenerator -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_setRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THShortTensor_setRNGState"
  c_THShortTensor_setRNGState :: Ptr CTHGenerator -> (Ptr CTHShortTensor) -> IO ()