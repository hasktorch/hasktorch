{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorMode where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCudaShortTensor_mode"
  c_mode :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCudaShortTensor_mode"
  p_mode :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())