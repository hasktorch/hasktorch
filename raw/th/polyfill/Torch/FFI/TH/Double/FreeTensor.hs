{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_DoubleTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THDoubleTensor -> IO ())

