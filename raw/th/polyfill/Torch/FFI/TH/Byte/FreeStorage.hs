{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_ByteStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THByteStorage -> IO ())

