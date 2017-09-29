{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorRandom (
    c_THShortTensor_random,
    c_THShortTensor_clampedRandom,
    c_THShortTensor_cappedRandom,
    c_THShortTensor_geometric,
    c_THShortTensor_bernoulli,
    c_THShortTensor_bernoulli_FloatTensor,
    c_THShortTensor_bernoulli_DoubleTensor) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_random : self _generator -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_random"
  c_THShortTensor_random :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THShortTensor_clampedRandom : self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_clampedRandom"
  c_THShortTensor_clampedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THShortTensor_cappedRandom : self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_cappedRandom"
  c_THShortTensor_cappedRandom :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THShortTensor_geometric : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_geometric"
  c_THShortTensor_geometric :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_bernoulli"
  c_THShortTensor_bernoulli :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THShortTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_bernoulli_FloatTensor"
  c_THShortTensor_bernoulli_FloatTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THShortTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THShortTensor_bernoulli_DoubleTensor"
  c_THShortTensor_bernoulli_DoubleTensor :: (Ptr CTHShortTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()