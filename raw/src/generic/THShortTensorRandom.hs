{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorRandom (
    c_THShortTensor_random,
    c_THShortTensor_clampedRandom,
    c_THShortTensor_cappedRandom,
    c_THShortTensor_geometric,
    c_THShortTensor_bernoulli,
    c_THShortTensor_bernoulli_FloatTensor,
    c_THShortTensor_bernoulli_DoubleTensor,
    c_THShortTensor_standard_gamma,
    p_THShortTensor_random,
    p_THShortTensor_clampedRandom,
    p_THShortTensor_cappedRandom,
    p_THShortTensor_geometric,
    p_THShortTensor_bernoulli,
    p_THShortTensor_bernoulli_FloatTensor,
    p_THShortTensor_bernoulli_DoubleTensor,
    p_THShortTensor_standard_gamma) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THShortTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THShortTensor_random"
  c_THShortTensor_random :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THShortTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THShortTensor_clampedRandom"
  c_THShortTensor_clampedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- |c_THShortTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THShortTensor_cappedRandom"
  c_THShortTensor_cappedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THShortTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_geometric"
  c_THShortTensor_geometric :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli"
  c_THShortTensor_bernoulli :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_FloatTensor"
  c_THShortTensor_bernoulli_FloatTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THShortTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_DoubleTensor"
  c_THShortTensor_bernoulli_DoubleTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THShortTensor_standard_gamma : self _generator alpha -> void
foreign import ccall "THTensorRandom.h THShortTensor_standard_gamma"
  c_THShortTensor_standard_gamma :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> (Ptr CTHShortTensor) -> IO ()

-- |p_THShortTensor_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THShortTensor_random"
  p_THShortTensor_random :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THShortTensor_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THShortTensor_clampedRandom"
  p_THShortTensor_clampedRandom :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_THShortTensor_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THShortTensor_cappedRandom"
  p_THShortTensor_cappedRandom :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THShortTensor_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_geometric"
  p_THShortTensor_geometric :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THShortTensor_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli"
  p_THShortTensor_bernoulli :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THShortTensor_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli_FloatTensor"
  p_THShortTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THShortTensor_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli_DoubleTensor"
  p_THShortTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_THShortTensor_standard_gamma : Pointer to function : self _generator alpha -> void
foreign import ccall "THTensorRandom.h &THShortTensor_standard_gamma"
  p_THShortTensor_standard_gamma :: FunPtr ((Ptr CTHShortTensor) -> Ptr CTHGenerator -> (Ptr CTHShortTensor) -> IO ())