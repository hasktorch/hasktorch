{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMathScan
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
foreign import ccall "THCTensorMathScan.h THDoubleTensor_cumsum"
  c_cumsum :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (())

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THDoubleTensor_cumprod"
  c_cumprod :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (())

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THDoubleTensor_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THDoubleTensor_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (()))