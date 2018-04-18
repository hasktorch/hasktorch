{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Sleep where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_THC_sleep :  state cycles -> void
foreign import ccall "THCSleep.h THC_sleep"
  c_THC_sleep :: Ptr C'THCState -> CLLong -> IO ()

-- | p_THC_sleep : Pointer to function : state cycles -> void
foreign import ccall "THCSleep.h &THC_sleep"
  p_THC_sleep :: FunPtr (Ptr C'THCState -> CLLong -> IO ())