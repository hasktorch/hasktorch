{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.CachingAllocator where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THCCachingAllocator_getBaseAllocation :  ptr size -> void *
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_getBaseAllocation"
  c_THCCachingAllocator_getBaseAllocation :: Ptr () -> Ptr CSize -> IO (Ptr ())

-- | c_THCCachingAllocator_recordStream :  ptr stream -> void
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_recordStream"
  c_THCCachingAllocator_recordStream :: Ptr () -> Ptr CTHCudaStream -> IO ()

-- | c_THCCachingAllocator_currentMemoryAllocated :  device -> uint64_t
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_currentMemoryAllocated"
  c_THCCachingAllocator_currentMemoryAllocated :: CInt -> IO CULong

-- | c_THCCachingAllocator_maxMemoryAllocated :  device -> uint64_t
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_maxMemoryAllocated"
  c_THCCachingAllocator_maxMemoryAllocated :: CInt -> IO CULong

-- | c_THCCachingAllocator_currentMemoryCached :  device -> uint64_t
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_currentMemoryCached"
  c_THCCachingAllocator_currentMemoryCached :: CInt -> IO CULong

-- | c_THCCachingAllocator_maxMemoryCached :  device -> uint64_t
foreign import ccall "THCCachingAllocator.h THCCachingAllocator_maxMemoryCached"
  c_THCCachingAllocator_maxMemoryCached :: CInt -> IO CULong

-- | p_THCCachingAllocator_getBaseAllocation : Pointer to function : ptr size -> void *
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_getBaseAllocation"
  p_THCCachingAllocator_getBaseAllocation :: FunPtr (Ptr () -> Ptr CSize -> IO (Ptr ()))

-- | p_THCCachingAllocator_recordStream : Pointer to function : ptr stream -> void
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_recordStream"
  p_THCCachingAllocator_recordStream :: FunPtr (Ptr () -> Ptr CTHCudaStream -> IO ())

-- | p_THCCachingAllocator_currentMemoryAllocated : Pointer to function : device -> uint64_t
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_currentMemoryAllocated"
  p_THCCachingAllocator_currentMemoryAllocated :: FunPtr (CInt -> IO CULong)

-- | p_THCCachingAllocator_maxMemoryAllocated : Pointer to function : device -> uint64_t
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_maxMemoryAllocated"
  p_THCCachingAllocator_maxMemoryAllocated :: FunPtr (CInt -> IO CULong)

-- | p_THCCachingAllocator_currentMemoryCached : Pointer to function : device -> uint64_t
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_currentMemoryCached"
  p_THCCachingAllocator_currentMemoryCached :: FunPtr (CInt -> IO CULong)

-- | p_THCCachingAllocator_maxMemoryCached : Pointer to function : device -> uint64_t
foreign import ccall "THCCachingAllocator.h &THCCachingAllocator_maxMemoryCached"
  p_THCCachingAllocator_maxMemoryCached :: FunPtr (CInt -> IO CULong)