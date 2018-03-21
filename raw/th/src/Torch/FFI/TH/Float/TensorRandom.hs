{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THFloatTensor_random"
  c_random :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_geometric"
  c_geometric :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli"
  c_bernoulli :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | c_uniform :  self _generator a b -> void
foreign import ccall "THTensorRandom.h THFloatTensor_uniform"
  c_uniform :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal"
  c_normal :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal_means :  self gen means stddev -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means"
  c_normal_means :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> CDouble -> IO ()

-- | c_normal_stddevs :  self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_stddevs"
  c_normal_stddevs :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> Ptr C'THFloatTensor -> IO ()

-- | c_normal_means_stddevs :  self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means_stddevs"
  c_normal_means_stddevs :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_exponential :  self _generator lambda -> void
foreign import ccall "THTensorRandom.h THFloatTensor_exponential"
  c_exponential :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_standard_gamma :  self _generator alpha -> void
foreign import ccall "THTensorRandom.h THFloatTensor_standard_gamma"
  c_standard_gamma :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_cauchy :  self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cauchy"
  c_cauchy :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | c_logNormal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_logNormal"
  c_logNormal :: Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | c_multinomial :  self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomial"
  c_multinomial :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ()

-- | c_multinomialAliasSetup :  prob_dist J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasSetup"
  c_multinomialAliasSetup :: Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | c_multinomialAliasDraw :  self _generator J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasDraw"
  c_multinomialAliasDraw :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_random"
  p_random :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())

-- | p_uniform : Pointer to function : self _generator a b -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_uniform"
  p_uniform :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal"
  p_normal :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal_means : Pointer to function : self gen means stddev -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means"
  p_normal_means :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> CDouble -> IO ())

-- | p_normal_stddevs : Pointer to function : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> Ptr C'THFloatTensor -> IO ())

-- | p_normal_means_stddevs : Pointer to function : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_exponential : Pointer to function : self _generator lambda -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_exponential"
  p_exponential :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_standard_gamma : Pointer to function : self _generator alpha -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_standard_gamma"
  p_standard_gamma :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_cauchy : Pointer to function : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cauchy"
  p_cauchy :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_logNormal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_logNormal"
  p_logNormal :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_multinomial : Pointer to function : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomial"
  p_multinomial :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> CInt -> CInt -> IO ())

-- | p_multinomialAliasSetup : Pointer to function : prob_dist J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_multinomialAliasDraw : Pointer to function : self _generator J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THLongTensor -> Ptr C'THFloatTensor -> IO ())