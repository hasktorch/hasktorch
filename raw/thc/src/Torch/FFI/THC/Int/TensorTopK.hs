{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THCIntTensor_topk"
  c_topk :: Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THCIntTensor_topk"
  p_topk :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaIntTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaIntTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())