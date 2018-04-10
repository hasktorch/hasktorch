{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_HalfTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THHalfTensor -> IO ())

