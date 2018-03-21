{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMathScan where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_cumsum :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCudaDoubleTensor_cumsum"
  c_cumsum :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCudaDoubleTensor_cumprod"
  c_cumprod :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCudaDoubleTensor_cumsum"
  p_cumsum :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCudaDoubleTensor_cumprod"
  p_cumprod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())