{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.CachingHostAllocator where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_THCCachingHostAllocator_emptyCache :   -> void
foreign import ccall "THCCachingHostAllocator.h THCCachingHostAllocator_emptyCache"
  c_THCCachingHostAllocator_emptyCache :: IO ()

-- | p_THCCachingHostAllocator_emptyCache : Pointer to function :  -> void
foreign import ccall "THCCachingHostAllocator.h &THCCachingHostAllocator_emptyCache"
  p_THCCachingHostAllocator_emptyCache :: FunPtr (IO ())