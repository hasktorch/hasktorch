{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Random where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THGenerator_new :   -> THGenerator *
foreign import ccall "THRandom.h THGenerator_new"
  c_THGenerator_new :: IO (Ptr C'THGenerator)

-- | c_THGenerator_copy :  self from -> THGenerator *
foreign import ccall "THRandom.h THGenerator_copy"
  c_THGenerator_copy :: Ptr C'THGenerator -> Ptr C'THGenerator -> IO (Ptr C'THGenerator)

-- | c_THGenerator_free :  gen -> void
foreign import ccall "THRandom.h THGenerator_free"
  c_THGenerator_free :: Ptr C'THGenerator -> IO ()

-- | c_THGenerator_isValid :  _generator -> int
foreign import ccall "THRandom.h THGenerator_isValid"
  c_THGenerator_isValid :: Ptr C'THGenerator -> IO CInt

-- | c_THRandom_seed :  _generator -> uint64_t
foreign import ccall "THRandom.h THRandom_seed"
  c_THRandom_seed :: Ptr C'THGenerator -> IO CULong

-- | c_THRandom_manualSeed :  _generator the_seed_ -> void
foreign import ccall "THRandom.h THRandom_manualSeed"
  c_THRandom_manualSeed :: Ptr C'THGenerator -> CULong -> IO ()

-- | c_THRandom_initialSeed :  _generator -> uint64_t
foreign import ccall "THRandom.h THRandom_initialSeed"
  c_THRandom_initialSeed :: Ptr C'THGenerator -> IO CULong

-- | c_THRandom_random :  _generator -> uint64_t
foreign import ccall "THRandom.h THRandom_random"
  c_THRandom_random :: Ptr C'THGenerator -> IO CULong

-- | c_THRandom_random64 :  _generator -> uint64_t
foreign import ccall "THRandom.h THRandom_random64"
  c_THRandom_random64 :: Ptr C'THGenerator -> IO CULong

-- | c_THRandom_uniform :  _generator a b -> double
foreign import ccall "THRandom.h THRandom_uniform"
  c_THRandom_uniform :: Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble

-- | c_THRandom_uniformFloat :  _generator a b -> float
foreign import ccall "THRandom.h THRandom_uniformFloat"
  c_THRandom_uniformFloat :: Ptr C'THGenerator -> CFloat -> CFloat -> IO CFloat

-- | c_THRandom_normal :  _generator mean stdv -> double
foreign import ccall "THRandom.h THRandom_normal"
  c_THRandom_normal :: Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble

-- | c_THRandom_exponential :  _generator lambda -> double
foreign import ccall "THRandom.h THRandom_exponential"
  c_THRandom_exponential :: Ptr C'THGenerator -> CDouble -> IO CDouble

-- | c_THRandom_standard_gamma :  _generator alpha -> double
foreign import ccall "THRandom.h THRandom_standard_gamma"
  c_THRandom_standard_gamma :: Ptr C'THGenerator -> CDouble -> IO CDouble

-- | c_THRandom_cauchy :  _generator median sigma -> double
foreign import ccall "THRandom.h THRandom_cauchy"
  c_THRandom_cauchy :: Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble

-- | c_THRandom_logNormal :  _generator mean stdv -> double
foreign import ccall "THRandom.h THRandom_logNormal"
  c_THRandom_logNormal :: Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble

-- | c_THRandom_geometric :  _generator p -> int
foreign import ccall "THRandom.h THRandom_geometric"
  c_THRandom_geometric :: Ptr C'THGenerator -> CDouble -> IO CInt

-- | c_THRandom_bernoulli :  _generator p -> int
foreign import ccall "THRandom.h THRandom_bernoulli"
  c_THRandom_bernoulli :: Ptr C'THGenerator -> CDouble -> IO CInt

-- | p_THGenerator_new : Pointer to function :  -> THGenerator *
foreign import ccall "THRandom.h &THGenerator_new"
  p_THGenerator_new :: FunPtr (IO (Ptr C'THGenerator))

-- | p_THGenerator_copy : Pointer to function : self from -> THGenerator *
foreign import ccall "THRandom.h &THGenerator_copy"
  p_THGenerator_copy :: FunPtr (Ptr C'THGenerator -> Ptr C'THGenerator -> IO (Ptr C'THGenerator))

-- | p_THGenerator_free : Pointer to function : gen -> void
foreign import ccall "THRandom.h &THGenerator_free"
  p_THGenerator_free :: FunPtr (Ptr C'THGenerator -> IO ())

-- | p_THGenerator_isValid : Pointer to function : _generator -> int
foreign import ccall "THRandom.h &THGenerator_isValid"
  p_THGenerator_isValid :: FunPtr (Ptr C'THGenerator -> IO CInt)

-- | p_THRandom_seed : Pointer to function : _generator -> uint64_t
foreign import ccall "THRandom.h &THRandom_seed"
  p_THRandom_seed :: FunPtr (Ptr C'THGenerator -> IO CULong)

-- | p_THRandom_manualSeed : Pointer to function : _generator the_seed_ -> void
foreign import ccall "THRandom.h &THRandom_manualSeed"
  p_THRandom_manualSeed :: FunPtr (Ptr C'THGenerator -> CULong -> IO ())

-- | p_THRandom_initialSeed : Pointer to function : _generator -> uint64_t
foreign import ccall "THRandom.h &THRandom_initialSeed"
  p_THRandom_initialSeed :: FunPtr (Ptr C'THGenerator -> IO CULong)

-- | p_THRandom_random : Pointer to function : _generator -> uint64_t
foreign import ccall "THRandom.h &THRandom_random"
  p_THRandom_random :: FunPtr (Ptr C'THGenerator -> IO CULong)

-- | p_THRandom_random64 : Pointer to function : _generator -> uint64_t
foreign import ccall "THRandom.h &THRandom_random64"
  p_THRandom_random64 :: FunPtr (Ptr C'THGenerator -> IO CULong)

-- | p_THRandom_uniform : Pointer to function : _generator a b -> double
foreign import ccall "THRandom.h &THRandom_uniform"
  p_THRandom_uniform :: FunPtr (Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble)

-- | p_THRandom_uniformFloat : Pointer to function : _generator a b -> float
foreign import ccall "THRandom.h &THRandom_uniformFloat"
  p_THRandom_uniformFloat :: FunPtr (Ptr C'THGenerator -> CFloat -> CFloat -> IO CFloat)

-- | p_THRandom_normal : Pointer to function : _generator mean stdv -> double
foreign import ccall "THRandom.h &THRandom_normal"
  p_THRandom_normal :: FunPtr (Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble)

-- | p_THRandom_exponential : Pointer to function : _generator lambda -> double
foreign import ccall "THRandom.h &THRandom_exponential"
  p_THRandom_exponential :: FunPtr (Ptr C'THGenerator -> CDouble -> IO CDouble)

-- | p_THRandom_standard_gamma : Pointer to function : _generator alpha -> double
foreign import ccall "THRandom.h &THRandom_standard_gamma"
  p_THRandom_standard_gamma :: FunPtr (Ptr C'THGenerator -> CDouble -> IO CDouble)

-- | p_THRandom_cauchy : Pointer to function : _generator median sigma -> double
foreign import ccall "THRandom.h &THRandom_cauchy"
  p_THRandom_cauchy :: FunPtr (Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble)

-- | p_THRandom_logNormal : Pointer to function : _generator mean stdv -> double
foreign import ccall "THRandom.h &THRandom_logNormal"
  p_THRandom_logNormal :: FunPtr (Ptr C'THGenerator -> CDouble -> CDouble -> IO CDouble)

-- | p_THRandom_geometric : Pointer to function : _generator p -> int
foreign import ccall "THRandom.h &THRandom_geometric"
  p_THRandom_geometric :: FunPtr (Ptr C'THGenerator -> CDouble -> IO CInt)

-- | p_THRandom_bernoulli : Pointer to function : _generator p -> int
foreign import ccall "THRandom.h &THRandom_bernoulli"
  p_THRandom_bernoulli :: FunPtr (Ptr C'THGenerator -> CDouble -> IO CInt)