{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMathScan
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
foreign import ccall "THCTensorMathScan.h THCharTensor_cumsum"
  c_cumsum :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THCharTensor_cumprod"
  c_cumprod :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCharTensor_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THCharTensor_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))