{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMode where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCudaIntTensor_mode"
  c_mode :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCudaIntTensor_mode"
  p_mode :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaIntTensor -> CInt -> CInt -> IO ())