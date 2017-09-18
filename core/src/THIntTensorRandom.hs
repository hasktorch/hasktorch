{-# LANGUAGE ForeignFunctionInterface#-}

module THIntTensorRandom (
    c_THIntTensor_random,
    c_THIntTensor_clampedRandom,
    c_THIntTensor_cappedRandom,
    c_THIntTensor_geometric,
    c_THIntTensor_bernoulli,
    c_THIntTensor_bernoulli_FloatTensor,
    c_THIntTensor_bernoulli_DoubleTensor,
    c_THIntTensor_uniform,
    c_THIntTensor_normal,
    c_THIntTensor_normal_means,
    c_THIntTensor_normal_stddevs,
    c_THIntTensor_normal_means_stddevs,
    c_THIntTensor_exponential,
    c_THIntTensor_cauchy,
    c_THIntTensor_logNormal,
    c_THIntTensor_multinomial,
    c_THIntTensor_multinomialAliasSetup,
    c_THIntTensor_multinomialAliasDraw,
    c_THIntTensor_getRNGState,
    c_THIntTensor_setRNGState) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THIntTensor_random"
  c_THIntTensor_random :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THIntTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THIntTensor_clampedRandom"
  c_THIntTensor_clampedRandom :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THIntTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THIntTensor_cappedRandom"
  c_THIntTensor_cappedRandom :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THIntTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_geometric"
  c_THIntTensor_geometric :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THIntTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli"
  c_THIntTensor_bernoulli :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THIntTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_FloatTensor"
  c_THIntTensor_bernoulli_FloatTensor :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THIntTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_DoubleTensor"
  c_THIntTensor_bernoulli_DoubleTensor :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THIntTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THIntTensor_uniform"
  c_THIntTensor_uniform :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THIntTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THIntTensor_normal"
  c_THIntTensor_normal :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THIntTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THIntTensor_normal_means"
  c_THIntTensor_normal_means :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> (Ptr CTHIntTensor) -> CDouble -> IO ()

-- |c_THIntTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THIntTensor_normal_stddevs"
  c_THIntTensor_normal_stddevs :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THIntTensor_normal_means_stddevs"
  c_THIntTensor_normal_means_stddevs :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THIntTensor_exponential"
  c_THIntTensor_exponential :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THIntTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THIntTensor_cauchy"
  c_THIntTensor_cauchy :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THIntTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THIntTensor_logNormal"
  c_THIntTensor_logNormal :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THIntTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THIntTensor_multinomial"
  c_THIntTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHIntTensor) -> CInt -> CInt -> IO ()

-- |c_THIntTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THIntTensor_multinomialAliasSetup"
  c_THIntTensor_multinomialAliasSetup :: (Ptr CTHIntTensor) -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THIntTensor_multinomialAliasDraw"
  c_THIntTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_getRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THIntTensor_getRNGState"
  c_THIntTensor_getRNGState :: Ptr CTHGenerator -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_setRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THIntTensor_setRNGState"
  c_THIntTensor_setRNGState :: Ptr CTHGenerator -> (Ptr CTHIntTensor) -> IO ()