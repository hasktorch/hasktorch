{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMode
  ( c_mode
  , p_mode
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THCDoubleTensor_mode"
  c_mode :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CInt -> CInt -> IO ()

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THCDoubleTensor_mode"
  p_mode :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CInt -> CInt -> IO ())