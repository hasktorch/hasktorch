{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Long.TensorMode
  ( c_mode
  , p_mode
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_mode :  state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h THLongTensor_mode"
  c_mode :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (())

-- | p_mode : Pointer to function : state values indices input dimension keepdim -> void
foreign import ccall "THCTensorMode.h &THLongTensor_mode"
  p_mode :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHLongTensor) -> CInt -> CInt -> IO (()))