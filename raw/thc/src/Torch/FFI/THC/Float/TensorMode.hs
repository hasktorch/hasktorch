{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMode where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCudaFloatTensor_mode"
  c_mode :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCudaFloatTensor_mode"
  p_mode :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> CInt -> CInt -> IO ())