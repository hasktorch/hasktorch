{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorRandom (
    c_THHalfTensor_random,
    c_THHalfTensor_clampedRandom,
    c_THHalfTensor_cappedRandom,
    c_THHalfTensor_geometric,
    c_THHalfTensor_bernoulli,
    c_THHalfTensor_bernoulli_FloatTensor,
    c_THHalfTensor_bernoulli_DoubleTensor) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_random : self _generator -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_random"
  c_THHalfTensor_random :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THHalfTensor_clampedRandom : self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_clampedRandom"
  c_THHalfTensor_clampedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THHalfTensor_cappedRandom : self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_cappedRandom"
  c_THHalfTensor_cappedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THHalfTensor_geometric : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_geometric"
  c_THHalfTensor_geometric :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_bernoulli"
  c_THHalfTensor_bernoulli :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_bernoulli_FloatTensor"
  c_THHalfTensor_bernoulli_FloatTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THHalfTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THHalfTensor_bernoulli_DoubleTensor"
  c_THHalfTensor_bernoulli_DoubleTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()