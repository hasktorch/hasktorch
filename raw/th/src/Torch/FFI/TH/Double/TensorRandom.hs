{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorRandom
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
foreign import ccall "THTensorRandom.h THDoubleTensor_random"
  c_random :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_geometric"
  c_geometric :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli"
  c_bernoulli :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- | c_uniform :  self _generator a b -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_uniform"
  c_uniform :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal"
  c_normal :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_normal_means :  self gen means stddev -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_means"
  c_normal_means :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> CDouble -> IO ()

-- | c_normal_stddevs :  self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_stddevs"
  c_normal_stddevs :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> Ptr CTHDoubleTensor -> IO ()

-- | c_normal_means_stddevs :  self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_means_stddevs"
  c_normal_means_stddevs :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_exponential :  self _generator lambda -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_exponential"
  c_exponential :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ()

-- | c_standard_gamma :  self _generator alpha -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_standard_gamma"
  c_standard_gamma :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- | c_cauchy :  self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_cauchy"
  c_cauchy :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_logNormal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_logNormal"
  c_logNormal :: Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ()

-- | c_multinomial :  self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomial"
  c_multinomial :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ()

-- | c_multinomialAliasSetup :  prob_dist J q -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomialAliasSetup"
  c_multinomialAliasSetup :: Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | c_multinomialAliasDraw :  self _generator J q -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomialAliasDraw"
  c_multinomialAliasDraw :: Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_random"
  p_random :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- | p_uniform : Pointer to function : self _generator a b -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_uniform"
  p_uniform :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal"
  p_normal :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal_means : Pointer to function : self gen means stddev -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_means"
  p_normal_means :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> CDouble -> IO ())

-- | p_normal_stddevs : Pointer to function : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> Ptr CTHDoubleTensor -> IO ())

-- | p_normal_means_stddevs : Pointer to function : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_exponential : Pointer to function : self _generator lambda -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_exponential"
  p_exponential :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> IO ())

-- | p_standard_gamma : Pointer to function : self _generator alpha -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_standard_gamma"
  p_standard_gamma :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- | p_cauchy : Pointer to function : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_cauchy"
  p_cauchy :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_logNormal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_logNormal"
  p_logNormal :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHGenerator -> CDouble -> CDouble -> IO ())

-- | p_multinomial : Pointer to function : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomial"
  p_multinomial :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> CInt -> CInt -> IO ())

-- | p_multinomialAliasSetup : Pointer to function : prob_dist J q -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr CTHDoubleTensor -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())

-- | p_multinomialAliasDraw : Pointer to function : self _generator J q -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr CTHLongTensor -> Ptr CTHGenerator -> Ptr CTHLongTensor -> Ptr CTHDoubleTensor -> IO ())