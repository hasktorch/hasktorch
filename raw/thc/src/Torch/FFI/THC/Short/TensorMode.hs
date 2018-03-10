{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorMode where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCShortTensor_mode"
  c_mode :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCShortTensor_mode"
  p_mode :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> CInt -> CInt -> IO ())