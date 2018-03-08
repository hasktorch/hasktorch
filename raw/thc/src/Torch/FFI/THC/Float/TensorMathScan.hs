{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorMathScan
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
foreign import ccall "THCTensorMathScan.h THFloatTensor_cumsum"
  c_cumsum :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | c_cumprod :  state self src dim -> void
foreign import ccall "THCTensorMathScan.h THFloatTensor_cumprod"
  c_cumprod :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (())

-- | p_cumsum : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THFloatTensor_cumsum"
  p_cumsum :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))

-- | p_cumprod : Pointer to function : state self src dim -> void
foreign import ccall "THCTensorMathScan.h &THFloatTensor_cumprod"
  p_cumprod :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHFloatTensor) -> CInt -> IO (()))