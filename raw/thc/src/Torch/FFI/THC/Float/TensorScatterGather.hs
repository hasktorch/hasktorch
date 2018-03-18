{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorScatterGather where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_gather :  state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_gather"
  c_gather :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatter"
  c_scatter :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatterAdd"
  c_scatterAdd :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCFloatTensor_scatterFill"
  c_scatterFill :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> CFloat -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_gather"
  p_gather :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatter"
  p_scatter :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCFloatTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CInt -> Ptr C'THCudaLongTensor -> CFloat -> IO ())