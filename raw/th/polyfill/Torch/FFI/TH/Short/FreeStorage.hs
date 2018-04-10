{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_ShortStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THShortStorage -> IO ())

