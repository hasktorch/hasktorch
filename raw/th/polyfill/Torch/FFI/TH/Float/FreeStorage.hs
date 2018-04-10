{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_FloatStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THFloatStorage -> IO ())

