{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.TensorRandom
  ( c_random
  , c_clampedRandom
  , c_cappedRandom
  , c_bernoulli
  , c_bernoulli_DoubleTensor
  , c_geometric
  , p_random
  , p_clampedRandom
  , p_cappedRandom
  , p_bernoulli
  , p_bernoulli_DoubleTensor
  , p_geometric
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_random :  state self -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_random"
  c_random :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_bernoulli"
  c_bernoulli :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCLongTensor_geometric"
  c_geometric :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CDouble -> IO ()

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_random"
  p_random :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCLongTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> CDouble -> IO ())