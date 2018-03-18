{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorRandom where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THShortTensor_random"
  c_random :: Ptr C'THShortTensor -> Ptr C'THGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THShortTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THShortTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THShortTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THShortTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_geometric"
  c_geometric :: Ptr C'THShortTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli"
  c_bernoulli :: Ptr C'THShortTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr C'THShortTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THShortTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THShortTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THShortTensor_random"
  p_random :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THShortTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THShortTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THShortTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THShortTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())