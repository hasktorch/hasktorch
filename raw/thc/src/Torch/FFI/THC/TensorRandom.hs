{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.TensorRandom
  ( c_THCRandom_init
  , c_THCRandom_shutdown
  , c_THCRandom_seed
  , c_THCRandom_seedAll
  , c_THCRandom_manualSeed
  , c_THCRandom_manualSeedAll
  , c_THCRandom_initialSeed
  , c_THCRandom_getRNGState
  , c_THCRandom_setRNGState
  , p_THCRandom_init
  , p_THCRandom_shutdown
  , p_THCRandom_seed
  , p_THCRandom_seedAll
  , p_THCRandom_manualSeed
  , p_THCRandom_manualSeedAll
  , p_THCRandom_initialSeed
  , p_THCRandom_getRNGState
  , p_THCRandom_setRNGState
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THCRandom_init :  state num_devices current_device -> void
foreign import ccall "THCTensorRandom.h THCRandom_init"
  c_THCRandom_init :: Ptr CTHCudaState -> CInt -> CInt -> IO ()

-- | c_THCRandom_shutdown :  state -> void
foreign import ccall "THCTensorRandom.h THCRandom_shutdown"
  c_THCRandom_shutdown :: Ptr CTHCudaState -> IO ()

-- | c_THCRandom_seed :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_seed"
  c_THCRandom_seed :: Ptr CTHCudaState -> IO CULong

-- | c_THCRandom_seedAll :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_seedAll"
  c_THCRandom_seedAll :: Ptr CTHCudaState -> IO CULong

-- | c_THCRandom_manualSeed :  state the_seed_ -> void
foreign import ccall "THCTensorRandom.h THCRandom_manualSeed"
  c_THCRandom_manualSeed :: Ptr CTHCudaState -> CULong -> IO ()

-- | c_THCRandom_manualSeedAll :  state the_seed_ -> void
foreign import ccall "THCTensorRandom.h THCRandom_manualSeedAll"
  c_THCRandom_manualSeedAll :: Ptr CTHCudaState -> CULong -> IO ()

-- | c_THCRandom_initialSeed :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_initialSeed"
  c_THCRandom_initialSeed :: Ptr CTHCudaState -> IO CULong

-- | c_THCRandom_getRNGState :  state rng_state -> void
foreign import ccall "THCTensorRandom.h THCRandom_getRNGState"
  c_THCRandom_getRNGState :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ()

-- | c_THCRandom_setRNGState :  state rng_state -> void
foreign import ccall "THCTensorRandom.h THCRandom_setRNGState"
  c_THCRandom_setRNGState :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ()

-- | p_THCRandom_init : Pointer to function : state num_devices current_device -> void
foreign import ccall "THCTensorRandom.h &THCRandom_init"
  p_THCRandom_init :: FunPtr (Ptr CTHCudaState -> CInt -> CInt -> IO ())

-- | p_THCRandom_shutdown : Pointer to function : state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_shutdown"
  p_THCRandom_shutdown :: FunPtr (Ptr CTHCudaState -> IO ())

-- | p_THCRandom_seed : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_seed"
  p_THCRandom_seed :: FunPtr (Ptr CTHCudaState -> IO CULong)

-- | p_THCRandom_seedAll : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_seedAll"
  p_THCRandom_seedAll :: FunPtr (Ptr CTHCudaState -> IO CULong)

-- | p_THCRandom_manualSeed : Pointer to function : state the_seed_ -> void
foreign import ccall "THCTensorRandom.h &THCRandom_manualSeed"
  p_THCRandom_manualSeed :: FunPtr (Ptr CTHCudaState -> CULong -> IO ())

-- | p_THCRandom_manualSeedAll : Pointer to function : state the_seed_ -> void
foreign import ccall "THCTensorRandom.h &THCRandom_manualSeedAll"
  p_THCRandom_manualSeedAll :: FunPtr (Ptr CTHCudaState -> CULong -> IO ())

-- | p_THCRandom_initialSeed : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_initialSeed"
  p_THCRandom_initialSeed :: FunPtr (Ptr CTHCudaState -> IO CULong)

-- | p_THCRandom_getRNGState : Pointer to function : state rng_state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_getRNGState"
  p_THCRandom_getRNGState :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ())

-- | p_THCRandom_setRNGState : Pointer to function : state rng_state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_setRNGState"
  p_THCRandom_setRNGState :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> IO ())