{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_DoubleStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THDoubleStorage -> IO ())

