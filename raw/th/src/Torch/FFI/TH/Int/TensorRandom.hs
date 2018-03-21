{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THIntTensor_random"
  c_random :: Ptr C'THIntTensor -> Ptr C'THGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THIntTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THIntTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_geometric"
  c_geometric :: Ptr C'THIntTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli"
  c_bernoulli :: Ptr C'THIntTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr C'THIntTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THIntTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THIntTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THIntTensor_random"
  p_random :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THIntTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THIntTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THIntTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THIntTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())