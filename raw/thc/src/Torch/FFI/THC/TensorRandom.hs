{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.TensorRandom
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
  c_THCRandom_init :: Ptr (CTHState) -> CInt -> CInt -> IO (())

-- | c_THCRandom_shutdown :  state -> void
foreign import ccall "THCTensorRandom.h THCRandom_shutdown"
  c_THCRandom_shutdown :: Ptr (CTHState) -> IO (())

-- | c_THCRandom_seed :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_seed"
  c_THCRandom_seed :: Ptr (CTHState) -> IO (CULong)

-- | c_THCRandom_seedAll :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_seedAll"
  c_THCRandom_seedAll :: Ptr (CTHState) -> IO (CULong)

-- | c_THCRandom_manualSeed :  state the_seed_ -> void
foreign import ccall "THCTensorRandom.h THCRandom_manualSeed"
  c_THCRandom_manualSeed :: Ptr (CTHState) -> CULong -> IO (())

-- | c_THCRandom_manualSeedAll :  state the_seed_ -> void
foreign import ccall "THCTensorRandom.h THCRandom_manualSeedAll"
  c_THCRandom_manualSeedAll :: Ptr (CTHState) -> CULong -> IO (())

-- | c_THCRandom_initialSeed :  state -> uint64_t
foreign import ccall "THCTensorRandom.h THCRandom_initialSeed"
  c_THCRandom_initialSeed :: Ptr (CTHState) -> IO (CULong)

-- | c_THCRandom_getRNGState :  state rng_state -> void
foreign import ccall "THCTensorRandom.h THCRandom_getRNGState"
  c_THCRandom_getRNGState :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> IO (())

-- | c_THCRandom_setRNGState :  state rng_state -> void
foreign import ccall "THCTensorRandom.h THCRandom_setRNGState"
  c_THCRandom_setRNGState :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> IO (())

-- | p_THCRandom_init : Pointer to function : state num_devices current_device -> void
foreign import ccall "THCTensorRandom.h &THCRandom_init"
  p_THCRandom_init :: FunPtr (Ptr (CTHState) -> CInt -> CInt -> IO (()))

-- | p_THCRandom_shutdown : Pointer to function : state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_shutdown"
  p_THCRandom_shutdown :: FunPtr (Ptr (CTHState) -> IO (()))

-- | p_THCRandom_seed : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_seed"
  p_THCRandom_seed :: FunPtr (Ptr (CTHState) -> IO (CULong))

-- | p_THCRandom_seedAll : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_seedAll"
  p_THCRandom_seedAll :: FunPtr (Ptr (CTHState) -> IO (CULong))

-- | p_THCRandom_manualSeed : Pointer to function : state the_seed_ -> void
foreign import ccall "THCTensorRandom.h &THCRandom_manualSeed"
  p_THCRandom_manualSeed :: FunPtr (Ptr (CTHState) -> CULong -> IO (()))

-- | p_THCRandom_manualSeedAll : Pointer to function : state the_seed_ -> void
foreign import ccall "THCTensorRandom.h &THCRandom_manualSeedAll"
  p_THCRandom_manualSeedAll :: FunPtr (Ptr (CTHState) -> CULong -> IO (()))

-- | p_THCRandom_initialSeed : Pointer to function : state -> uint64_t
foreign import ccall "THCTensorRandom.h &THCRandom_initialSeed"
  p_THCRandom_initialSeed :: FunPtr (Ptr (CTHState) -> IO (CULong))

-- | p_THCRandom_getRNGState : Pointer to function : state rng_state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_getRNGState"
  p_THCRandom_getRNGState :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_THCRandom_setRNGState : Pointer to function : state rng_state -> void
foreign import ccall "THCTensorRandom.h &THCRandom_setRNGState"
  p_THCRandom_setRNGState :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> IO (()))