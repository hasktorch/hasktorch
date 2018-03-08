{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Half.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THHalfTensor_topk"
  c_topk :: Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THHalfTensor_topk"
  p_topk :: FunPtr (Ptr (CTHState) -> Ptr (CTHHalfTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHHalfTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))