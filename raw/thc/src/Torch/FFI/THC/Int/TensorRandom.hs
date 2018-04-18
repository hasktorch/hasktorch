{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorRandom where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_random"
  c_random :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_clampedRandom"
  c_clampedRandom :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_cappedRandom"
  c_cappedRandom :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_bernoulli"
  c_bernoulli :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCudaIntTensor_geometric"
  c_geometric :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CDouble -> IO ()

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_random"
  p_random :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCudaIntTensor_geometric"
  p_geometric :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CDouble -> IO ())