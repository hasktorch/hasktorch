{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_LongTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THLongTensor -> IO ())

