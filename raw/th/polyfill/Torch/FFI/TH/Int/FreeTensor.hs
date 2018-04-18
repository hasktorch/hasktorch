{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_IntTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THIntTensor -> IO ())

