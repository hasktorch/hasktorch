{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_FloatTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THFloatTensor -> IO ())

