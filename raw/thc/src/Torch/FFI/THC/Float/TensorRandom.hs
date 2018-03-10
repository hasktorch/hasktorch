{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorRandom where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_uniform :  state self a b -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_uniform"
  c_uniform :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_rand :  state r_ size -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_rand"
  c_rand :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_randn :  state r_ size -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_randn"
  c_randn :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_normal :  state self mean stdv -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_normal"
  c_normal :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_normal_means :  state self means stddev -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_normal_means"
  c_normal_means :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CDouble -> IO ()

-- | c_normal_stddevs :  state self mean stddevs -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_normal_stddevs"
  c_normal_stddevs :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_normal_means_stddevs :  state self means stddevs -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_normal_means_stddevs"
  c_normal_means_stddevs :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_logNormal :  state self mean stdv -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_logNormal"
  c_logNormal :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_exponential :  state self lambda -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_exponential"
  c_exponential :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ()

-- | c_cauchy :  state self median sigma -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_cauchy"
  c_cauchy :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ()

-- | c_multinomial :  state self prob_dist n_sample with_replacement -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_multinomial"
  c_multinomial :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ()

-- | c_multinomialAliasSetup :  state probs J q -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_multinomialAliasSetup"
  c_multinomialAliasSetup :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_multinomialAliasDraw :  state self _J _q -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_multinomialAliasDraw"
  c_multinomialAliasDraw :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_random"
  c_random :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_bernoulli"
  c_bernoulli :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCFloatTensor_geometric"
  c_geometric :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ()

-- | p_uniform : Pointer to function : state self a b -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_uniform"
  p_uniform :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_rand : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_rand"
  p_rand :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_randn : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_randn"
  p_randn :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_normal : Pointer to function : state self mean stdv -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_normal"
  p_normal :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_normal_means : Pointer to function : state self means stddev -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_normal_means"
  p_normal_means :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> CDouble -> IO ())

-- | p_normal_stddevs : Pointer to function : state self mean stddevs -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_normal_means_stddevs : Pointer to function : state self means stddevs -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_logNormal : Pointer to function : state self mean stdv -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_logNormal"
  p_logNormal :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_exponential : Pointer to function : state self lambda -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_exponential"
  p_exponential :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ())

-- | p_cauchy : Pointer to function : state self median sigma -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_cauchy"
  p_cauchy :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> CDouble -> IO ())

-- | p_multinomial : Pointer to function : state self prob_dist n_sample with_replacement -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_multinomial"
  p_multinomial :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> CInt -> CInt -> IO ())

-- | p_multinomialAliasSetup : Pointer to function : state probs J q -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_multinomialAliasDraw : Pointer to function : state self _J _q -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_random"
  p_random :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCFloatTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CDouble -> IO ())