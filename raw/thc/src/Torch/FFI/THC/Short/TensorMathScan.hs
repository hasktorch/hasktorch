{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorMathScan
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
foreign import ccall "THCTensorMathScan.h THCShortTensor_cumsum"
  c_cumsum :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> IO ()

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCShortTensor_cumprod"
  c_cumprod :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> IO ()

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCShortTensor_cumsum"
  p_cumsum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> IO ())

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCShortTensor_cumprod"
  p_cumprod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> IO ())