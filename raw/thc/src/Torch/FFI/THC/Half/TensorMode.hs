{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMode
  ( c_mode
  , p_mode
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCHalfTensor_mode"
  c_mode :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCHalfTensor_mode"
  p_mode :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())