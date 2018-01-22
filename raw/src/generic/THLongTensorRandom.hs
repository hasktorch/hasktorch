{-# LANGUAGE ForeignFunctionInterface #-}

module THLongTensorRandom (
    c_THLongTensor_random,
    c_THLongTensor_clampedRandom,
    c_THLongTensor_cappedRandom,
    c_THLongTensor_geometric,
    c_THLongTensor_bernoulli,
    c_THLongTensor_bernoulli_FloatTensor,
    c_THLongTensor_bernoulli_DoubleTensor,
    c_THLongTensor_bernoulli_Tensor,
    p_THLongTensor_random,
    p_THLongTensor_clampedRandom,
    p_THLongTensor_cappedRandom,
    p_THLongTensor_geometric,
    p_THLongTensor_bernoulli,
    p_THLongTensor_bernoulli_FloatTensor,
    p_THLongTensor_bernoulli_DoubleTensor,
    p_THLongTensor_bernoulli_Tensor) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THLongTensor_random : self _generator -> void
foreign import ccall "THTensorRandom.h THLongTensor_random"
  c_THLongTensor_random :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> IO ()

-- |c_THLongTensor_clampedRandom : self _generator min max -> void
foreign import ccall "THTensorRandom.h THLongTensor_clampedRandom"
  c_THLongTensor_clampedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ()

-- |c_THLongTensor_cappedRandom : self _generator max -> void
foreign import ccall "THTensorRandom.h THLongTensor_cappedRandom"
  c_THLongTensor_cappedRandom :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLLong -> IO ()

-- |c_THLongTensor_geometric : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_geometric"
  c_THLongTensor_geometric :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli"
  c_THLongTensor_bernoulli :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ()

-- |c_THLongTensor_bernoulli_FloatTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_FloatTensor"
  c_THLongTensor_bernoulli_FloatTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ()

-- |c_THLongTensor_bernoulli_DoubleTensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_DoubleTensor"
  c_THLongTensor_bernoulli_DoubleTensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ()

-- |c_THLongTensor_bernoulli_Tensor : self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_Tensor"
  c_THLongTensor_bernoulli_Tensor :: (Ptr CTHLongTensor) -> Ptr CTHGenerator -> (Ptr CTHLongTensor) -> IO ()

-- |p_THLongTensor_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THLongTensor_random"
  p_THLongTensor_random :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> IO ())

-- |p_THLongTensor_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THLongTensor_clampedRandom"
  p_THLongTensor_clampedRandom :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLLong -> CLLong -> IO ())

-- |p_THLongTensor_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THLongTensor_cappedRandom"
  p_THLongTensor_cappedRandom :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> CLLong -> IO ())

-- |p_THLongTensor_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_geometric"
  p_THLongTensor_geometric :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THLongTensor_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli"
  p_THLongTensor_bernoulli :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> CDouble -> IO ())

-- |p_THLongTensor_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli_FloatTensor"
  p_THLongTensor_bernoulli_FloatTensor :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHFloatTensor -> IO ())

-- |p_THLongTensor_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli_DoubleTensor"
  p_THLongTensor_bernoulli_DoubleTensor :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> Ptr CTHDoubleTensor -> IO ())

-- |p_THLongTensor_bernoulli_Tensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli_Tensor"
  p_THLongTensor_bernoulli_Tensor :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHGenerator -> (Ptr CTHLongTensor) -> IO ())