{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorRandom where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  self _generator -> void
foreign import ccall "THTensorRandom.h THLongTensor_random"
  c_random :: Ptr C'THLongTensor -> Ptr C'THGenerator -> IO ()

-- | c_clampedRandom :  self _generator min max -> void
foreign import ccall "THTensorRandom.h THLongTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  self _generator max -> void
foreign import ccall "THTensorRandom.h THLongTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> IO ()

-- | c_geometric :  self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_geometric"
  c_geometric :: Ptr C'THLongTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli :  self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli"
  c_bernoulli :: Ptr C'THLongTensor -> Ptr C'THGenerator -> CDouble -> IO ()

-- | c_bernoulli_FloatTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_FloatTensor"
  c_bernoulli_FloatTensor :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ()

-- | c_bernoulli_DoubleTensor :  self _generator p -> void
foreign import ccall "THTensorRandom.h THLongTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ()

-- | p_random : Pointer to function : self _generator -> void
foreign import ccall "THTensorRandom.h &THLongTensor_random"
  p_random :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> IO ())

-- | p_clampedRandom : Pointer to function : self _generator min max -> void
foreign import ccall "THTensorRandom.h &THLongTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : self _generator max -> void
foreign import ccall "THTensorRandom.h &THLongTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> CLLong -> IO ())

-- | p_geometric : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> CDouble -> IO ())

-- | p_bernoulli_FloatTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli_FloatTensor"
  p_bernoulli_FloatTensor :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THFloatTensor -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : self _generator p -> void
foreign import ccall "THTensorRandom.h &THLongTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THLongTensor -> Ptr C'THGenerator -> Ptr C'THDoubleTensor -> IO ())