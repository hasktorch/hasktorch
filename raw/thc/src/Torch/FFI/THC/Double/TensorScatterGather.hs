{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorScatterGather where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_gather :  state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h THCudaDoubleTensor_gather"
  c_gather :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCudaDoubleTensor_scatter"
  c_scatter :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCudaDoubleTensor_scatterAdd"
  c_scatterAdd :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCudaDoubleTensor_scatterFill"
  c_scatterFill :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> CDouble -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCudaDoubleTensor_gather"
  p_gather :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCudaDoubleTensor_scatter"
  p_scatter :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCudaDoubleTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCudaDoubleTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CInt -> Ptr C'THCudaLongTensor -> CDouble -> IO ())