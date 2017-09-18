{-# LANGUAGE ForeignFunctionInterface#-}

module THLongTensorRandom (
    c_THLongTensor_random,
    c_THLongTensor_clampedRandom,
    c_THLongTensor_cappedRandom,
    c_THLongTensor_geometric,
    c_THLongTensor_bernoulli,
    c_THLongTensor_bernoulli_FloatTensor,
    c_THLongTensor_bernoulli_DoubleTensor,
    c_THLongTensor_uniform,
    c_THLongTensor_normal,
    c_THLongTensor_normal_means,
    c_THLongTensor_normal_stddevs,
    c_THLongTensor_normal_means_stddevs,
    c_THLongTensor_exponential,
    c_THLongTensor_cauchy,
    c_THLongTensor_logNormal,
    c_THLongTensor_multinomial,
    c_THLongTensor_multinomialAliasSetup,
    c_THLongTensor_multinomialAliasDraw,
    c_THLongTensor_getRNGState,
    c_THLongTensor_setRNGState) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THLongTensor_random"
  c_THLongTensor_random :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THLongTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THLongTensor_clampedRandom"
  c_THLongTensor_clampedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THLongTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THLongTensor_cappedRandom"
  c_THLongTensor_cappedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THLongTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_geometric"
  c_THLongTensor_geometric :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli"
  c_THLongTensor_bernoulli :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_FloatTensor"
  c_THLongTensor_bernoulli_FloatTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THLongTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_DoubleTensor"
  c_THLongTensor_bernoulli_DoubleTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THLongTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THLongTensor_uniform"
  c_THLongTensor_uniform :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THLongTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THLongTensor_normal"
  c_THLongTensor_normal :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THLongTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THLongTensor_normal_means"
  c_THLongTensor_normal_means :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> (Ptr CTHLongTensor) -> CDouble -> IO ()

-- |c_THLongTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THLongTensor_normal_stddevs"
  c_THLongTensor_normal_stddevs :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THLongTensor_normal_means_stddevs"
  c_THLongTensor_normal_means_stddevs :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THLongTensor_exponential"
  c_THLongTensor_exponential :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THLongTensor_cauchy"
  c_THLongTensor_cauchy :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THLongTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THLongTensor_logNormal"
  c_THLongTensor_logNormal :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THLongTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THLongTensor_multinomial"
  c_THLongTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHLongTensor) -> CInt -> CInt -> IO ()

-- |c_THLongTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THLongTensor_multinomialAliasSetup"
  c_THLongTensor_multinomialAliasSetup :: (Ptr CTHLongTensor) -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THLongTensor_multinomialAliasDraw"
  c_THLongTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_getRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THLongTensor_getRNGState"
  c_THLongTensor_getRNGState :: Ptr CTHGenerator -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_setRNGState : _generator self -> void
foreign import ccall "THTensorRandom.h THLongTensor_setRNGState"
  c_THLongTensor_setRNGState :: Ptr CTHGenerator -> (Ptr CTHLongTensor) -> IO ()