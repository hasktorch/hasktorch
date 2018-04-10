{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.FreeTensor where

import Foreign
import Data.Word
import Torch.Types.TH

foreign import ccall "&free_CharTensor"
  p_free :: FunPtr (Ptr C'THState -> Ptr C'THCharTensor -> IO ())

