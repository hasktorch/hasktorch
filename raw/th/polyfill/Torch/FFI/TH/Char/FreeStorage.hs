{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.FreeStorage where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_CharStorage"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THCharStorage -> IO ())

