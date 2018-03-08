{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THDoubleTensor_topk"
  c_topk :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THDoubleTensor_topk"
  p_topk :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))