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
    c_THFloatTensor_multinomialAliasDraw,
    p_THFloatTensor_random,
    p_THFloatTensor_clampedRandom,
    p_THFloatTensor_cappedRandom,
    p_THFloatTensor_geometric,
    p_THFloatTensor_bernoulli,
    p_THFloatTensor_bernoulli_FloatTensor,
    p_THFloatTensor_bernoulli_DoubleTensor,
    p_THFloatTensor_uniform,
    p_THFloatTensor_normal,
    p_THFloatTensor_normal_means,
    p_THFloatTensor_normal_stddevs,
    p_THFloatTensor_normal_means_stddevs,
    p_THFloatTensor_exponential,
    p_THFloatTensor_cauchy,
    p_THFloatTensor_logNormal,
    p_THFloatTensor_multinomial,
    p_THFloatTensor_multinomialAliasSetup,
    p_THFloatTensor_multinomialAliasDraw) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THFloatTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THFloatTensor_random"
  c_THFloatTensor_random :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THFloatTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_clampedRandom"
  c_THFloatTensor_clampedRandom :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- |c_THFloatTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cappedRandom"
  c_THFloatTensor_cappedRandom :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THFloatTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_geometric"
  c_THFloatTensor_geometric :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli"
  c_THFloatTensor_bernoulli :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_FloatTensor"
  c_THFloatTensor_bernoulli_FloatTensor :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THFloatTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_DoubleTensor"
  c_THFloatTensor_bernoulli_DoubleTensor :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THFloatTensor_uniform : self _generator a b -> void
foreign import ccall "THTensorRandom.h THFloatTensor_uniform"
  c_THFloatTensor_uniform :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_normal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal"
  c_THFloatTensor_normal :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_normal_means : self gen means stddev -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means"
  c_THFloatTensor_normal_means :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CDouble -> IO ()

-- |c_THFloatTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_stddevs"
  c_THFloatTensor_normal_stddevs :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means_stddevs"
  c_THFloatTensor_normal_means_stddevs :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_exponential : self _generator lambda -> void
foreign import ccall "THTensorRandom.h THFloatTensor_exponential"
  c_THFloatTensor_exponential :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THFloatTensor_cauchy : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cauchy"
  c_THFloatTensor_cauchy :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_logNormal : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_logNormal"
  c_THFloatTensor_logNormal :: (Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THFloatTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomial"
  c_THFloatTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ()

-- |c_THFloatTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasSetup"
  c_THFloatTensor_multinomialAliasSetup :: (Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasDraw"
  c_THFloatTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ()

-- |p_THFloatTensor_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_random"
  p_THFloatTensor_random :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THFloatTensor_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_clampedRandom"
  p_THFloatTensor_clampedRandom :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_THFloatTensor_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cappedRandom"
  p_THFloatTensor_cappedRandom :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THFloatTensor_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_geometric"
  p_THFloatTensor_geometric :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THFloatTensor_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli"
  p_THFloatTensor_bernoulli :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THFloatTensor_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_FloatTensor"
  p_THFloatTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THFloatTensor_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_DoubleTensor"
  p_THFloatTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_THFloatTensor_uniform : Pointer to function : self _generator a b -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_uniform"
  p_THFloatTensor_uniform :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_normal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal"
  p_THFloatTensor_normal :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_normal_means : Pointer to function : self gen means stddev -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means"
  p_THFloatTensor_normal_means :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CDouble -> IO ())

-- |p_THFloatTensor_normal_stddevs : Pointer to function : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_stddevs"
  p_THFloatTensor_normal_stddevs :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_normal_means_stddevs : Pointer to function : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means_stddevs"
  p_THFloatTensor_normal_means_stddevs :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_exponential : Pointer to function : self _generator lambda -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_exponential"
  p_THFloatTensor_exponential :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THFloatTensor_cauchy : Pointer to function : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cauchy"
  p_THFloatTensor_cauchy :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_logNormal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_logNormal"
  p_THFloatTensor_logNormal :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THFloatTensor_multinomial : Pointer to function : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomial"
  p_THFloatTensor_multinomial :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHFloatTensor) -> CInt -> CInt -> IO ())

-- |p_THFloatTensor_multinomialAliasSetup : Pointer to function : prob_dist J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasSetup"
  p_THFloatTensor_multinomialAliasSetup :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_multinomialAliasDraw : Pointer to function : self _generator J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasDraw"
  p_THFloatTensor_multinomialAliasDraw :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHFloatTensor) -> IO ())