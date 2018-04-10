{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_HalfStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THHalfStorage -> IO ())

