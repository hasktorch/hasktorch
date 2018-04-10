{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_ShortTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THShortTensor -> IO ())

