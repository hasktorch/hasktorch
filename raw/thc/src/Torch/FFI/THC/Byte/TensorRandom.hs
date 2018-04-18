{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Byte.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_random"
  c_random :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_bernoulli"
  c_bernoulli :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaByteTensor_geometric"
  c_geometric :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> CDouble -> IO ()

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_random"
  p_random :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaByteTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> CDouble -> IO ())