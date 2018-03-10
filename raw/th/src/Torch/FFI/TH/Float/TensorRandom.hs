{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_geometric
  , c_bernoulli
  , c_bernoulli_FloatTensor
  , c_bernoulli_DoubleTensor
  , c_uniform
  , c_normal
  , c_normal_means
  , c_normal_stddevs
  , c_normal_means_stddevs
  , c_exponential
  , c_standard_gamma
  , c_cauchy
  , c_logNormal
  , c_multinomial
  , c_multinomialAliasSetup
  , c_multinomialAliasDraw
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_geometric
  , p_bernoulli
  , p_bernoulli_FloatTensor
  , p_bernoulli_DoubleTensor
  , p_uniform
  , p_normal
  , p_normal_means
  , p_normal_stddevs
  , p_normal_means_stddevs
  , p_exponential
  , p_standard_gamma
  , p_cauchy
  , p_logNormal
  , p_multinomial
  , p_multinomialAliasSetup
  , p_multinomialAliasDraw
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THFloatTensor_random"
  c_random :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_geometric"
  c_geometric :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli"
  c_bernoulli :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THFloatTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- | c_uniform :  self _generator a b -> void
foreign import ccall "THTensorRandom.h THFloatTensor_uniform"
  c_uniform :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal"
  c_normal :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal_means :  self gen means stddev -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means"
  c_normal_means :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> CDouble -> IO ()

-- | c_normal_stddevs :  self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_stddevs"
  c_normal_stddevs :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> Ptr CTHFloatTensor -> IO ()

-- | c_normal_means_stddevs :  self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THFloatTensor_normal_means_stddevs"
  c_normal_means_stddevs :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_exponential :  self _generator lambda -> void
foreign import ccall "THTensorRandom.h THFloatTensor_exponential"
  c_exponential :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_standard_gamma :  self _generator alpha -> void
foreign import ccall "THTensorRandom.h THFloatTensor_standard_gamma"
  c_standard_gamma :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- | c_cauchy :  self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THFloatTensor_cauchy"
  c_cauchy :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_logNormal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THFloatTensor_logNormal"
  c_logNormal :: Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_multinomial :  self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomial"
  c_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> CInt -> CInt -> IO ()

-- | c_multinomialAliasSetup :  prob_dist J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasSetup"
  c_multinomialAliasSetup :: Ptr CTHFloatTensor -> Ptr CTHLongTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_multinomialAliasDraw :  self _generator J q -> void
foreign import ccall "THTensorRandom.h THFloatTensor_multinomialAliasDraw"
  c_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr CTHFloatTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_random"
  p_random :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- | p_uniform : Pointer to function : self _generator a b -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_uniform"
  p_uniform :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal"
  p_normal :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal_means : Pointer to function : self gen means stddev -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means"
  p_normal_means :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> CDouble -> IO ())

-- | p_normal_stddevs : Pointer to function : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> Ptr CTHFloatTensor -> IO ())

-- | p_normal_means_stddevs : Pointer to function : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_exponential : Pointer to function : self _generator lambda -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_exponential"
  p_exponential :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_standard_gamma : Pointer to function : self _generator alpha -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_standard_gamma"
  p_standard_gamma :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- | p_cauchy : Pointer to function : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_cauchy"
  p_cauchy :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_logNormal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_logNormal"
  p_logNormal :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_multinomial : Pointer to function : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomial"
  p_multinomial :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> CInt -> CInt -> IO ())

-- | p_multinomialAliasSetup : Pointer to function : prob_dist J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHLongTensor -> Ptr CTHFloatTensor -> IO ())

-- | p_multinomialAliasDraw : Pointer to function : self _generator J q -> void
foreign import ccall "THTensorRandom.h &THFloatTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr CTHFloatTensor -> IO ())