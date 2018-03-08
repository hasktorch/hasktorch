{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THIntTensor_topk"
  c_topk :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THIntTensor_topk"
  p_topk :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHIntTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))