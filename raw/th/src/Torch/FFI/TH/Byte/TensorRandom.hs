{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorRandom where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THByteTensor_random"
  c_random :: Ptr C'THByteTensor -> Ptr C'THGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THByteTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THByteTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_geometric"
  c_geometric :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli"
  c_bernoulli :: Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THByteTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | c_getRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_getRNGState"
  c_getRNGState :: Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()

-- | c_setRNGState :  _generator self -> void
foreign import ccall "THTensorRandom.h THByteTensor_setRNGState"
  c_setRNGState :: Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THByteTensor_random"
  p_random :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THByteTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THByteTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THByteTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THByteTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())

-- | p_getRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &THByteTensor_getRNGState"
  p_getRNGState :: FunPtr (Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ())

-- | p_setRNGState : Pointer to function : _generator self -> void
foreign import ccall "THTensorRandom.h &THByteTensor_setRNGState"
  p_setRNGState :: FunPtr (Ptr C'THGenerator -> Ptr C'THByteTensor -> IO ())