{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.CachingHostAllocator
  ( c_THCCachingHostAllocator_emptyCache
  , p_THCCachingHostAllocator_emptyCache
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THCCachingHostAllocator_emptyCache :   -> void
foreign import ccall "THCCachingHostAllocator.h THCCachingHostAllocator_emptyCache"
  c_THCCachingHostAllocator_emptyCache :: IO (())

-- | p_THCCachingHostAllocator_emptyCache : Pointer to function :  -> void
foreign import ccall "THCCachingHostAllocator.h &THCCachingHostAllocator_emptyCache"
  p_THCCachingHostAllocator_emptyCache :: FunPtr (IO (()))