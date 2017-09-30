{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleTensorRandom (
    c_THDoubleTensor_random,
    c_THDoubleTensor_clampedRandom,
    c_THDoubleTensor_cappedRandom,
    c_THDoubleTensor_geometric,
    c_THDoubleTensor_bernoulli,
    c_THDoubleTensor_bernoulli_FloatTensor,
    c_THDoubleTensor_bernoulli_DoubleTensor,
    c_THDoubleTensor_uniform,
    c_THDoubleTensor_normal,
    c_THDoubleTensor_normal_means,
    c_THDoubleTensor_normal_stddevs,
    c_THDoubleTensor_normal_means_stddevs,
    c_THDoubleTensor_exponential,
    c_THDoubleTensor_cauchy,
    c_THDoubleTensor_logNormal,
    c_THDoubleTensor_multinomial,
    c_THDoubleTensor_multinomialAliasSetup,
    c_THDoubleTensor_multinomialAliasDraw,
    p_THDoubleTensor_random,
    p_THDoubleTensor_clampedRandom,
    p_THDoubleTensor_cappedRandom,
    p_THDoubleTensor_geometric,
    p_THDoubleTensor_bernoulli,
    p_THDoubleTensor_bernoulli_FloatTensor,
    p_THDoubleTensor_bernoulli_DoubleTensor,
    p_THDoubleTensor_uniform,
    p_THDoubleTensor_normal,
    p_THDoubleTensor_normal_means,
    p_THDoubleTensor_normal_stddevs,
    p_THDoubleTensor_normal_means_stddevs,
    p_THDoubleTensor_exponential,
    p_THDoubleTensor_cauchy,
    p_THDoubleTensor_logNormal,
    p_THDoubleTensor_multinomial,
    p_THDoubleTensor_multinomialAliasSetup,
    p_THDoubleTensor_multinomialAliasDraw) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_random : self _generator -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_random"
  c_THDoubleTensor_random :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THDoubleTensor_clampedRandom : self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_clampedRandom"
  c_THDoubleTensor_clampedRandom :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THDoubleTensor_cappedRandom : self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_cappedRandom"
  c_THDoubleTensor_cappedRandom :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THDoubleTensor_geometric : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_geometric"
  c_THDoubleTensor_geometric :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THDoubleTensor_bernoulli : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_bernoulli"
  c_THDoubleTensor_bernoulli :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THDoubleTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_bernoulli_FloatTensor"
  c_THDoubleTensor_bernoulli_FloatTensor :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THDoubleTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_bernoulli_DoubleTensor"
  c_THDoubleTensor_bernoulli_DoubleTensor :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THDoubleTensor_uniform : self _generator a b -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_uniform"
  c_THDoubleTensor_uniform :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_normal : self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_normal"
  c_THDoubleTensor_normal :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_normal_means : self gen means stddev -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_normal_means"
  c_THDoubleTensor_normal_means :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> CDouble -> IO ()

-- |c_THDoubleTensor_normal_stddevs : self gen mean stddevs -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_normal_stddevs"
  c_THDoubleTensor_normal_stddevs :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_normal_means_stddevs : self gen means stddevs -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_normal_means_stddevs"
  c_THDoubleTensor_normal_means_stddevs :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_exponential : self _generator lambda -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_exponential"
  c_THDoubleTensor_exponential :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THDoubleTensor_cauchy : self _generator median sigma -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_cauchy"
  c_THDoubleTensor_cauchy :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_logNormal : self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_logNormal"
  c_THDoubleTensor_logNormal :: (Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- |c_THDoubleTensor_multinomial : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_multinomial"
  c_THDoubleTensor_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ()

-- |c_THDoubleTensor_multinomialAliasSetup : prob_dist J q -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_multinomialAliasSetup"
  c_THDoubleTensor_multinomialAliasSetup :: (Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_multinomialAliasDraw : self _generator J q -> void
foreign import ccall unsafe "THTensorRandom.h THDoubleTensor_multinomialAliasDraw"
  c_THDoubleTensor_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ()

-- |p_THDoubleTensor_random : Pointer to self _generator -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_random"
  p_THDoubleTensor_random :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THDoubleTensor_clampedRandom : Pointer to self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_clampedRandom"
  p_THDoubleTensor_clampedRandom :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ())

-- |p_THDoubleTensor_cappedRandom : Pointer to self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_cappedRandom"
  p_THDoubleTensor_cappedRandom :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CLong -> IO ())

-- |p_THDoubleTensor_geometric : Pointer to self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_geometric"
  p_THDoubleTensor_geometric :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THDoubleTensor_bernoulli : Pointer to self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_bernoulli"
  p_THDoubleTensor_bernoulli :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THDoubleTensor_bernoulli_FloatTensor : Pointer to self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_bernoulli_FloatTensor"
  p_THDoubleTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THDoubleTensor_bernoulli_DoubleTensor : Pointer to self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_bernoulli_DoubleTensor"
  p_THDoubleTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_THDoubleTensor_uniform : Pointer to self _generator a b -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_uniform"
  p_THDoubleTensor_uniform :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_normal : Pointer to self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_normal"
  p_THDoubleTensor_normal :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_normal_means : Pointer to self gen means stddev -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_normal_means"
  p_THDoubleTensor_normal_means :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> CDouble -> IO ())

-- |p_THDoubleTensor_normal_stddevs : Pointer to self gen mean stddevs -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_normal_stddevs"
  p_THDoubleTensor_normal_stddevs :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_normal_means_stddevs : Pointer to self gen means stddevs -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_normal_means_stddevs"
  p_THDoubleTensor_normal_means_stddevs :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_exponential : Pointer to self _generator lambda -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_exponential"
  p_THDoubleTensor_exponential :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THDoubleTensor_cauchy : Pointer to self _generator median sigma -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_cauchy"
  p_THDoubleTensor_cauchy :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_logNormal : Pointer to self _generator mean stdv -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_logNormal"
  p_THDoubleTensor_logNormal :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- |p_THDoubleTensor_multinomial : Pointer to self _generator prob_dist n_sample with_replacement -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_multinomial"
  p_THDoubleTensor_multinomial :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> (Ptr CTHDoubleTensor) -> CInt -> CInt -> IO ())

-- |p_THDoubleTensor_multinomialAliasSetup : Pointer to prob_dist J q -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_multinomialAliasSetup"
  p_THDoubleTensor_multinomialAliasSetup :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_multinomialAliasDraw : Pointer to self _generator J q -> void
foreign import ccall unsafe "THTensorRandom.h &THDoubleTensor_multinomialAliasDraw"
  p_THDoubleTensor_multinomialAliasDraw :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> (Ptr CTHDoubleTensor) -> IO ())