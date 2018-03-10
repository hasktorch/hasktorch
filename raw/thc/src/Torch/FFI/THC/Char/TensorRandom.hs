{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorRandom
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
foreign import ccall "THCTensorRandom.h THCCharTensor_random"
  c_random :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO ()

-- | c_clampedRandom :  state self min max -> void
foreign import ccall "THCTensorRandom.h THCCharTensor_clampedRandom"
  c_clampedRandom :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CLLong -> CLLong -> IO ()

-- | c_cappedRandom :  state self max -> void
foreign import ccall "THCTensorRandom.h THCCharTensor_cappedRandom"
  c_cappedRandom :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CLLong -> IO ()

-- | c_bernoulli :  state self p -> void
foreign import ccall "THCTensorRandom.h THCCharTensor_bernoulli"
  c_bernoulli :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CDouble -> IO ()

-- | c_bernoulli_DoubleTensor :  state self p -> void
foreign import ccall "THCTensorRandom.h THCCharTensor_bernoulli_DoubleTensor"
  c_bernoulli_DoubleTensor :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_geometric :  state self p -> void
foreign import ccall "THCTensorRandom.h THCCharTensor_geometric"
  c_geometric :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CDouble -> IO ()

-- | p_random : Pointer to function : state self -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_random"
  p_random :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO ())

-- | p_clampedRandom : Pointer to function : state self min max -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_clampedRandom"
  p_clampedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CLLong -> CLLong -> IO ())

-- | p_cappedRandom : Pointer to function : state self max -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_cappedRandom"
  p_cappedRandom :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CLLong -> IO ())

-- | p_bernoulli : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_bernoulli"
  p_bernoulli :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CDouble -> IO ())

-- | p_bernoulli_DoubleTensor : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_bernoulli_DoubleTensor"
  p_bernoulli_DoubleTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_geometric : Pointer to function : state self p -> void
foreign import ccall "THCTensorRandom.h &THCCharTensor_geometric"
  p_geometric :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CDouble -> IO ())