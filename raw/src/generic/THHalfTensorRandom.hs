{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorRandom (
    c_THHalfTensor_random,
    c_THHalfTensor_clampedRandom,
    c_THHalfTensor_cappedRandom,
    c_THHalfTensor_geometric,
    c_THHalfTensor_bernoulli,
    c_THHalfTensor_bernoulli_FloatTensor,
    c_THHalfTensor_bernoulli_DoubleTensor,
    c_THHalfTensor_bernoulli_Tensor,
    p_THHalfTensor_random,
    p_THHalfTensor_clampedRandom,
    p_THHalfTensor_cappedRandom,
    p_THHalfTensor_geometric,
    p_THHalfTensor_bernoulli,
    p_THHalfTensor_bernoulli_FloatTensor,
    p_THHalfTensor_bernoulli_DoubleTensor,
    p_THHalfTensor_bernoulli_Tensor) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THHalfTensor_random"
  c_THHalfTensor_random :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THHalfTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_clampedRandom"
  c_THHalfTensor_clampedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- |c_THHalfTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THHalfTensor_cappedRandom"
  c_THHalfTensor_cappedRandom :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THHalfTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_geometric"
  c_THHalfTensor_geometric :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli"
  c_THHalfTensor_bernoulli :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THHalfTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_FloatTensor"
  c_THHalfTensor_bernoulli_FloatTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THHalfTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_DoubleTensor"
  c_THHalfTensor_bernoulli_DoubleTensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THHalfTensor_bernoulli_Tensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THHalfTensor_bernoulli_Tensor"
  c_THHalfTensor_bernoulli_Tensor :: (Ptr CTHHalfTensor) -> Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> IO ()

-- |p_THHalfTensor_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_random"
  p_THHalfTensor_random :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THHalfTensor_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_clampedRandom"
  p_THHalfTensor_clampedRandom :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_THHalfTensor_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_cappedRandom"
  p_THHalfTensor_cappedRandom :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THHalfTensor_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_geometric"
  p_THHalfTensor_geometric :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THHalfTensor_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli"
  p_THHalfTensor_bernoulli :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THHalfTensor_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli_FloatTensor"
  p_THHalfTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THHalfTensor_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli_DoubleTensor"
  p_THHalfTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_THHalfTensor_bernoulli_Tensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THHalfTensor_bernoulli_Tensor"
  p_THHalfTensor_bernoulli_Tensor :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHGenerator -> (Ptr CTHHalfTensor) -> IO ())