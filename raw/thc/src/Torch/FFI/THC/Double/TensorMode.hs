{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMode where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCDoubleTensor_mode"
  c_mode :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCDoubleTensor_mode"
  p_mode :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> CInt -> CInt -> IO ())