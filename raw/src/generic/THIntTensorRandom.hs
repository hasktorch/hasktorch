{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorRandom (
    c_THIntTensor_random,
    c_THIntTensor_clampedRandom,
    c_THIntTensor_cappedRandom,
    c_THIntTensor_geometric,
    c_THIntTensor_bernoulli,
    c_THIntTensor_bernoulli_FloatTensor,
    c_THIntTensor_bernoulli_DoubleTensor,
    p_THIntTensor_random,
    p_THIntTensor_clampedRandom,
    p_THIntTensor_cappedRandom,
    p_THIntTensor_geometric,
    p_THIntTensor_bernoulli,
    p_THIntTensor_bernoulli_FloatTensor,
    p_THIntTensor_bernoulli_DoubleTensor) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THIntTensor_random"
  c_THIntTensor_random :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THIntTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THIntTensor_clampedRandom"
  c_THIntTensor_clampedRandom :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- |c_THIntTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THIntTensor_cappedRandom"
  c_THIntTensor_cappedRandom :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THIntTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_geometric"
  c_THIntTensor_geometric :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THIntTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli"
  c_THIntTensor_bernoulli :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THIntTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_FloatTensor"
  c_THIntTensor_bernoulli_FloatTensor :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THIntTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_DoubleTensor"
  c_THIntTensor_bernoulli_DoubleTensor :: (Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |p_THIntTensor_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THIntTensor_random"
  p_THIntTensor_random :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THIntTensor_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THIntTensor_clampedRandom"
  p_THIntTensor_clampedRandom :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_THIntTensor_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THIntTensor_cappedRandom"
  p_THIntTensor_cappedRandom :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THIntTensor_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_geometric"
  p_THIntTensor_geometric :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THIntTensor_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli"
  p_THIntTensor_bernoulli :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THIntTensor_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli_FloatTensor"
  p_THIntTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THIntTensor_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli_DoubleTensor"
  p_THIntTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())