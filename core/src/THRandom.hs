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
    c_THRandom_bernoulli) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THGenerator_new :  -> THGenerator *
foreign import ccall "THRandom.h THGenerator_new"
  c_THGenerator_new :: IO (Ptr CTHGenerator)

-- |c_THGenerator_copy : self from -> THGenerator *
foreign import ccall "THRandom.h THGenerator_copy"
  c_THGenerator_copy :: Ptr CTHGenerator -> Ptr CTHGenerator -> IO (Ptr CTHGenerator)

-- |c_THGenerator_free : gen -> void
foreign import ccall "THRandom.h THGenerator_free"
  c_THGenerator_free :: Ptr CTHGenerator -> IO ()

-- |c_THGenerator_isValid : _generator -> int
foreign import ccall "THRandom.h THGenerator_isValid"
  c_THGenerator_isValid :: Ptr CTHGenerator -> CInt

-- |c_THRandom_seed : _generator -> long
foreign import ccall "THRandom.h THRandom_seed"
  c_THRandom_seed :: Ptr CTHGenerator -> CLong

-- |c_THRandom_manualSeed : _generator the_seed_ -> void
foreign import ccall "THRandom.h THRandom_manualSeed"
  c_THRandom_manualSeed :: Ptr CTHGenerator -> CLong -> IO ()

-- |c_THRandom_initialSeed : _generator -> long
foreign import ccall "THRandom.h THRandom_initialSeed"
  c_THRandom_initialSeed :: Ptr CTHGenerator -> CLong

-- |c_THRandom_random : _generator -> long
foreign import ccall "THRandom.h THRandom_random"
  c_THRandom_random :: Ptr CTHGenerator -> CLong

-- |c_THRandom_uniform : _generator a b -> double
foreign import ccall "THRandom.h THRandom_uniform"
  c_THRandom_uniform :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_normal : _generator mean stdv -> double
foreign import ccall "THRandom.h THRandom_normal"
  c_THRandom_normal :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_exponential : _generator lambda -> double
foreign import ccall "THRandom.h THRandom_exponential"
  c_THRandom_exponential :: Ptr CTHGenerator -> CDouble -> CDouble

-- |c_THRandom_cauchy : _generator median sigma -> double
foreign import ccall "THRandom.h THRandom_cauchy"
  c_THRandom_cauchy :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_logNormal : _generator mean stdv -> double
foreign import ccall "THRandom.h THRandom_logNormal"
  c_THRandom_logNormal :: Ptr CTHGenerator -> CDouble -> CDouble -> CDouble

-- |c_THRandom_geometric : _generator p -> int
foreign import ccall "THRandom.h THRandom_geometric"
  c_THRandom_geometric :: Ptr CTHGenerator -> CDouble -> CInt

-- |c_THRandom_bernoulli : _generator p -> int
foreign import ccall "THRandom.h THRandom_bernoulli"
  c_THRandom_bernoulli :: Ptr CTHGenerator -> CDouble -> CInt