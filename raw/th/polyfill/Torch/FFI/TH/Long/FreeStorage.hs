{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_LongStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THLongStorage -> IO ())

