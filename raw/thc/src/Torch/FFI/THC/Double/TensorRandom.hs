{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorRandom
  ( c_uniform
  , c_rand
  , c_randn
  , c_normal
  , c_normal_means
  , c_normal_stddevs
  , c_normal_means_stddevs
  , c_logNormal
  , c_exponential
  , c_cauchy
  , c_multinomial
  , c_multinomialAliasSetup
  , c_multinomialAliasDraw
  , c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_bernoulli
  , c_bernoulli_DoubleTensor
  , c_geometric
  , p_uniform
  , p_rand
  , p_randn
  , p_normal
  , p_normal_means
  , p_normal_stddevs
  , p_normal_means_stddevs
  , p_logNormal
  , p_exponential
  , p_cauchy
  , p_multinomial
  , p_multinomialAliasSetup
  , p_multinomialAliasDraw
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_bernoulli
  , p_bernoulli_DoubleTensor
  , p_geometric
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_uniform :  state self a b -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_uniform"
  c_uniform :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (())

-- | c_rand :  state r_ size -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_rand"
  c_rand :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_randn :  state r_ size -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_randn"
  c_randn :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_normal :  state self mean stdv -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_normal"
  c_normal :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (())

-- | c_normal_means :  state self means stddev -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_normal_means"
  c_normal_means :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_normal_stddevs :  state self mean stddevs -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_normal_stddevs"
  c_normal_stddevs :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_normal_means_stddevs :  state self means stddevs -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_normal_means_stddevs"
  c_normal_means_stddevs :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_logNormal :  state self mean stdv -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_logNormal"
  c_logNormal :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (())

-- | c_exponential :  state self lambda -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_exponential"
  c_exponential :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_cauchy :  state self median sigma -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_cauchy"
  c_cauchy :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (())

-- | c_multinomial :  state self prob_dist n_sample with_replacement -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_multinomial"
  c_multinomial :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> CInt -> CInt -> IO (())

-- | c_multinomialAliasSetup :  state probs J q -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_multinomialAliasSetup"
  c_multinomialAliasSetup :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_multinomialAliasDraw :  state self _J _q -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_multinomialAliasDraw"
  c_multinomialAliasDraw :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_random"
  c_random :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_clampedRandom"
  c_clampedRandom :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> CLLong -> IO (())

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_cappedRandom"
  c_cappedRandom :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (())

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_bernoulli"
  c_bernoulli :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THDoubleTensor_geometric"
  c_geometric :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | p_uniform : Pointer to function : state self a b -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_uniform"
  p_uniform :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (()))

-- | p_rand : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_rand"
  p_rand :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_randn : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_randn"
  p_randn :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_normal : Pointer to function : state self mean stdv -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_normal"
  p_normal :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (()))

-- | p_normal_means : Pointer to function : state self means stddev -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_normal_means"
  p_normal_means :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_normal_stddevs : Pointer to function : state self mean stddevs -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_normal_means_stddevs : Pointer to function : state self means stddevs -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_logNormal : Pointer to function : state self mean stdv -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_logNormal"
  p_logNormal :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (()))

-- | p_exponential : Pointer to function : state self lambda -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_exponential"
  p_exponential :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_cauchy : Pointer to function : state self median sigma -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_cauchy"
  p_cauchy :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> IO (()))

-- | p_multinomial : Pointer to function : state self prob_dist n_sample with_replacement -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_multinomial"
  p_multinomial :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> CInt -> CInt -> IO (()))

-- | p_multinomialAliasSetup : Pointer to function : state probs J q -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_multinomialAliasDraw : Pointer to function : state self _J _q -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_random"
  p_random :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> CLLong -> IO (()))

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (()))

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THDoubleTensor_geometric"
  p_geometric :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))