{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMathScan
  ( c_cumsum
  , c_cumprod
  , p_cumsum
  , p_cumprod
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_cumsum :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCIntTensor_cumsum"
  c_cumsum :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> IO ()

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCIntTensor_cumprod"
  c_cumprod :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> IO ()

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCIntTensor_cumsum"
  p_cumsum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCIntTensor_cumprod"
  p_cumprod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> IO ())