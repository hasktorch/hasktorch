{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Sleep where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_THC_sleep :  state cycles -> void
foreign import ccall "THCSleep.h THC_sleep"
  c_THC_sleep :: Ptr CTHCudaState -> CLLong -> IO ()

-- | p_THC_sleep : Pointer to function : state cycles -> void
foreign import ccall "THCSleep.h &THC_sleep"
  p_THC_sleep :: FunPtr (Ptr CTHCudaState -> CLLong -> IO ())