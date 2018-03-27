{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_ByteTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THByteTensor -> IO ())

