{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_random"
  c_random_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> IO ()

-- | alias of c_random_ with unused argument (for CTHState) to unify backpack signatures.
c_random :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> IO ()
c_random = const c_random_

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_clampedRandom"
  c_clampedRandom_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | alias of c_clampedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_clampedRandom :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()
c_clampedRandom = const c_clampedRandom_

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_cappedRandom"
  c_cappedRandom_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | alias of c_cappedRandom_ with unused argument (for CTHState) to unify backpack signatures.
c_cappedRandom :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ()
c_cappedRandom = const c_cappedRandom_

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_geometric"
  c_geometric_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_geometric_ with unused argument (for CTHState) to unify backpack signatures.
c_geometric :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_geometric = const c_geometric_

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli"
  c_bernoulli_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_bernoulli_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_bernoulli = const c_bernoulli_

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_bernoulli_FloatTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_FloatTensor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()
c_bernoulli_FloatTensor = const c_bernoulli_FloatTensor_

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_bernoulli_DoubleTensor_ with unused argument (for CTHState) to unify backpack signatures.
c_bernoulli_DoubleTensor :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()
c_bernoulli_DoubleTensor = const c_bernoulli_DoubleTensor_

-- | c_uniform :  self _generator a b -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_uniform"
  c_uniform_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | alias of c_uniform_ with unused argument (for CTHState) to unify backpack signatures.
c_uniform :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()
c_uniform = const c_uniform_

-- | c_normal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal"
  c_normal_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | alias of c_normal_ with unused argument (for CTHState) to unify backpack signatures.
c_normal :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()
c_normal = const c_normal_

-- | c_normal_means :  self gen means stddev -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_means"
  c_normal_means_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CDouble -> IO ()

-- | alias of c_normal_means_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_means :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CDouble -> IO ()
c_normal_means = const c_normal_means_

-- | c_normal_stddevs :  self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_stddevs"
  c_normal_stddevs_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_normal_stddevs_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_stddevs :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> Ptr C'THDoubleTensor -> IO ()
c_normal_stddevs = const c_normal_stddevs_

-- | c_normal_means_stddevs :  self gen means stddevs -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_normal_means_stddevs"
  c_normal_means_stddevs_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_normal_means_stddevs_ with unused argument (for CTHState) to unify backpack signatures.
c_normal_means_stddevs :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()
c_normal_means_stddevs = const c_normal_means_stddevs_

-- | c_exponential :  self _generator lambda -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_exponential"
  c_exponential_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | alias of c_exponential_ with unused argument (for CTHState) to unify backpack signatures.
c_exponential :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ()
c_exponential = const c_exponential_

-- | c_standard_gamma :  self _generator alpha -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_standard_gamma"
  c_standard_gamma_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_standard_gamma_ with unused argument (for CTHState) to unify backpack signatures.
c_standard_gamma :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()
c_standard_gamma = const c_standard_gamma_

-- | c_cauchy :  self _generator median sigma -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_cauchy"
  c_cauchy_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | alias of c_cauchy_ with unused argument (for CTHState) to unify backpack signatures.
c_cauchy :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()
c_cauchy = const c_cauchy_

-- | c_logNormal :  self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_logNormal"
  c_logNormal_ :: Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()

-- | alias of c_logNormal_ with unused argument (for CTHState) to unify backpack signatures.
c_logNormal :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ()
c_logNormal = const c_logNormal_

-- | c_multinomial :  self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomial"
  c_multinomial_ :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()

-- | alias of c_multinomial_ with unused argument (for CTHState) to unify backpack signatures.
c_multinomial :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ()
c_multinomial = const c_multinomial_

-- | c_multinomialAliasSetup :  prob_dist J q -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomialAliasSetup"
  c_multinomialAliasSetup_ :: Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_multinomialAliasSetup_ with unused argument (for CTHState) to unify backpack signatures.
c_multinomialAliasSetup :: Ptr C'THState -> Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_multinomialAliasSetup = const c_multinomialAliasSetup_

-- | c_multinomialAliasDraw :  self _generator J q -> void
foreign import ccall "THTensorRandom.h THDoubleTensor_multinomialAliasDraw"
  c_multinomialAliasDraw_ :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_multinomialAliasDraw_ with unused argument (for CTHState) to unify backpack signatures.
c_multinomialAliasDraw :: Ptr C'THState -> Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ()
c_multinomialAliasDraw = const c_multinomialAliasDraw_

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_random"
  p_random :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())

-- | p_uniform : Pointer to function : self _generator a b -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_uniform"
  p_uniform :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal"
  p_normal :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_normal_means : Pointer to function : self gen means stddev -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_means"
  p_normal_means :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CDouble -> IO ())

-- | p_normal_stddevs : Pointer to function : self gen mean stddevs -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_stddevs"
  p_normal_stddevs :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> Ptr C'THDoubleTensor -> IO ())

-- | p_normal_means_stddevs : Pointer to function : self gen means stddevs -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_normal_means_stddevs"
  p_normal_means_stddevs :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_exponential : Pointer to function : self _generator lambda -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_exponential"
  p_exponential :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_standard_gamma : Pointer to function : self _generator alpha -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_standard_gamma"
  p_standard_gamma :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())

-- | p_cauchy : Pointer to function : self _generator median sigma -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_cauchy"
  p_cauchy :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_logNormal : Pointer to function : self _generator mean stdv -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_logNormal"
  p_logNormal :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THGenerator -> CDouble -> CDouble -> IO ())

-- | p_multinomial : Pointer to function : self _generator prob_dist n_sample with_replacement -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomial"
  p_multinomial :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> CInt -> CInt -> IO ())

-- | p_multinomialAliasSetup : Pointer to function : prob_dist J q -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomialAliasSetup"
  p_multinomialAliasSetup :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())

-- | p_multinomialAliasDraw : Pointer to function : self _generator J q -> void
foreign import ccall "THTensorRandom.h &THDoubleTensor_multinomialAliasDraw"
  p_multinomialAliasDraw :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THLongTensor -> Ptr C'THDoubleTensor -> IO ())