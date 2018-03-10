{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorTopK where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THCDoubleTensor_topk"
  c_topk :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ()

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THCDoubleTensor_topk"
  p_topk :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> CInt -> CInt -> CInt -> IO ())