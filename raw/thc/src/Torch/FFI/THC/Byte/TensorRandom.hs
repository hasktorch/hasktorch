{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.TensorRandom where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_random"
  c_random :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_bernoulli"
  c_bernoulli :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCByteTensor_geometric"
  c_geometric :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CDouble -> IO ()

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_random"
  p_random :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCByteTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> CDouble -> IO ())