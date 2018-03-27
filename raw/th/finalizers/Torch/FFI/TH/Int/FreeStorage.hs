{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_IntStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THIntStorage -> IO ())

