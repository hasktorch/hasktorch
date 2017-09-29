{-# LANGUAGE ForeignFunctionInterface #-}

module THLongTensorRandom (
    c_THLongTensor_random,
    c_THLongTensor_clampedRandom,
    c_THLongTensor_cappedRandom,
    c_THLongTensor_geometric,
    c_THLongTensor_bernoulli,
    c_THLongTensor_bernoulli_FloatTensor,
    c_THLongTensor_bernoulli_DoubleTensor) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_random : self _generator -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_random"
  c_THLongTensor_random :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THLongTensor_clampedRandom : self _generator min max -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_clampedRandom"
  c_THLongTensor_clampedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> CLong -> IO ()

-- |c_THLongTensor_cappedRandom : self _generator max -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_cappedRandom"
  c_THLongTensor_cappedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLong -> IO ()

-- |c_THLongTensor_geometric : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_geometric"
  c_THLongTensor_geometric :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_bernoulli"
  c_THLongTensor_bernoulli :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_bernoulli_FloatTensor"
  c_THLongTensor_bernoulli_FloatTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THLongTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall unsafe "THTensorRandom.h THLongTensor_bernoulli_DoubleTensor"
  c_THLongTensor_bernoulli_DoubleTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()