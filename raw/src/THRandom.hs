{-# LANGUAGE ForeignFunctionInterface #-}

module THRandom (
    c_THGenerator_new,
    c_THGenerator_copy,
    c_THGenerator_free,
    c_THGenerator_isValid,
    c_THRandom_seed,
    c_THRandom_manualSeed,
    c_THRandom_initialSeed,
    c_THRandom_random,
    c_THRandom_uniform,
    c_THRandom_normal,
    c_THRandom_exponential,
    c_THRandom_cauchy,
    c_THRandom_logNormal,
    c_THRandom_geometric,
    c_THRandom_bernoulli,
    p_THGenerator_new,
    p_THGenerator_copy,
    p_THGenerator_free,
    p_THGenerator_isValid,
    p_THRandom_seed,
    p_THRandom_manualSeed,
    p_THRandom_initialSeed,
    p_THRandom_random,
    p_THRandom_uniform,
    p_THRandom_normal,
    p_THRandom_exponential,
    p_THRandom_cauchy,
    p_THRandom_logNormal,
    p_THRandom_geometric,
    p_THRandom_bernoulli) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THGenerator_new :  -> THGenerator *
foreign import ccall unsafe "THRandom.h THGenerator_new"
  c_THGenerator_new :: IO (Ptr CTHGenerator)

-- |c_THGenerator_copy : self from -> THGenerator *
foreign import ccall unsafe "THRandom.h THGenerator_copy"
  c_THGenerator_copy :: Ptr CTHGenerator -> Ptr CTHGenerator -> IO (Ptr CTHGenerator)

-- |c_THGenerator_free : gen -> void
foreign import ccall unsafe "THRandom.h THGenerator_free"
  c_THGenerator_free :: Ptr CTHGenerator -> IO ()

-- |c_THGenerator_isValid : _generator -> int
foreign import ccall unsafe "THRandom.h THGenerator_isValid"
  c_THGenerator_isValid :: Ptr CTHGenerator -> CInt

-- |c_THRandom_seed : _generator -> long
foreign import ccall unsafe "THRandom.h THRandom_seed"
  c_THRandom_seed :: Ptr CTHGenerator -> CLong

-- |c_THRandom_manualSeed : _generator the_seed_ -> void
foreign import ccall unsafe "THRandom.h THRandom_manualSeed"
  c_THRandom_manualSeed :: Ptr CTHGenerator -> CLong -> IO ()

-- |c_THRandom_initialSeed : _generator -> long
foreign import ccall unsafe "THRandom.h THRandom_initialSeed"
  c_THRandom_initialSeed :: Ptr CTHGenerator -> CLong

-- |c_THRandom_random : _generator -> long
foreign import ccall unsafe "THRandom.h THRandom_random"
  c_THRandom_random :: Ptr CTHGenerator -> CLong

-- |c_THRandom_uniform : _generator a b -> double
foreign import ccall unsafe "THRandom.h THRandom_uniform"
  c_THRandom_uniform :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_normal : _generator mean stdv -> double
foreign import ccall unsafe "THRandom.h THRandom_normal"
  c_THRandom_normal :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_exponential : _generator lambda -> double
foreign import ccall unsafe "THRandom.h THRandom_exponential"
  c_THRandom_exponential :: Ptr CTHGenerator -> CDouble -> CDouble

-- |c_THRandom_cauchy : _generator median sigma -> double
foreign import ccall unsafe "THRandom.h THRandom_cauchy"
  c_THRandom_cauchy :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_logNormal : _generator mean stdv -> double
foreign import ccall unsafe "THRandom.h THRandom_logNormal"
  c_THRandom_logNormal :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_geometric : _generator p -> int
foreign import ccall unsafe "THRandom.h THRandom_geometric"
  c_THRandom_geometric :: Ptr CTHGenerator -> CDouble -> CInt

-- |c_THRandom_bernoulli : _generator p -> int
foreign import ccall unsafe "THRandom.h THRandom_bernoulli"
  c_THRandom_bernoulli :: Ptr CTHGenerator -> CDouble -> CInt

-- |p_THGenerator_new : Pointer to function  -> THGenerator *
foreign import ccall unsafe "THRandom.h &THGenerator_new"
  p_THGenerator_new :: FunPtr (IO (Ptr CTHGenerator))

-- |p_THGenerator_copy : Pointer to function self from -> THGenerator *
foreign import ccall unsafe "THRandom.h &THGenerator_copy"
  p_THGenerator_copy :: FunPtr (Ptr CTHGenerator -> Ptr CTHGenerator -> IO (Ptr CTHGenerator))

-- |p_THGenerator_free : Pointer to function gen -> void
foreign import ccall unsafe "THRandom.h &THGenerator_free"
  p_THGenerator_free :: FunPtr (Ptr CTHGenerator -> IO ())

-- |p_THGenerator_isValid : Pointer to function _generator -> int
foreign import ccall unsafe "THRandom.h &THGenerator_isValid"
  p_THGenerator_isValid :: FunPtr (Ptr CTHGenerator -> CInt)

-- |p_THRandom_seed : Pointer to function _generator -> long
foreign import ccall unsafe "THRandom.h &THRandom_seed"
  p_THRandom_seed :: FunPtr (Ptr CTHGenerator -> CLong)

-- |p_THRandom_manualSeed : Pointer to function _generator the_seed_ -> void
foreign import ccall unsafe "THRandom.h &THRandom_manualSeed"
  p_THRandom_manualSeed :: FunPtr (Ptr CTHGenerator -> CLong -> IO ())

-- |p_THRandom_initialSeed : Pointer to function _generator -> long
foreign import ccall unsafe "THRandom.h &THRandom_initialSeed"
  p_THRandom_initialSeed :: FunPtr (Ptr CTHGenerator -> CLong)

-- |p_THRandom_random : Pointer to function _generator -> long
foreign import ccall unsafe "THRandom.h &THRandom_random"
  p_THRandom_random :: FunPtr (Ptr CTHGenerator -> CLong)

-- |p_THRandom_uniform : Pointer to function _generator a b -> double
foreign import ccall unsafe "THRandom.h &THRandom_uniform"
  p_THRandom_uniform :: FunPtr (Ptr CTHGenerator -> CDouble -> CDouble -> CDouble)

-- |p_THRandom_normal : Pointer to function _generator mean stdv -> double
foreign import ccall unsafe "THRandom.h &THRandom_normal"
  p_THRandom_normal :: FunPtr (Ptr CTHGenerator -> CDouble -> CDouble -> CDouble)

-- |p_THRandom_exponential : Pointer to function _generator lambda -> double
foreign import ccall unsafe "THRandom.h &THRandom_exponential"
  p_THRandom_exponential :: FunPtr (Ptr CTHGenerator -> CDouble -> CDouble)

-- |p_THRandom_cauchy : Pointer to function _generator median sigma -> double
foreign import ccall unsafe "THRandom.h &THRandom_cauchy"
  p_THRandom_cauchy :: FunPtr (Ptr CTHGenerator -> CDouble -> CDouble -> CDouble)

-- |p_THRandom_logNormal : Pointer to function _generator mean stdv -> double
foreign import ccall unsafe "THRandom.h &THRandom_logNormal"
  p_THRandom_logNormal :: FunPtr (Ptr CTHGenerator -> CDouble -> CDouble -> CDouble)

-- |p_THRandom_geometric : Pointer to function _generator p -> int
foreign import ccall unsafe "THRandom.h &THRandom_geometric"
  p_THRandom_geometric :: FunPtr (Ptr CTHGenerator -> CDouble -> CInt)

-- |p_THRandom_bernoulli : Pointer to function _generator p -> int
foreign import ccall unsafe "THRandom.h &THRandom_bernoulli"
  p_THRandom_bernoulli :: FunPtr (Ptr CTHGenerator -> CDouble -> CInt)