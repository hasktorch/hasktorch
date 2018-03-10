{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorScatterGather where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gather :  state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h THCDoubleTensor_gather"
  c_gather :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCDoubleTensor_scatter"
  c_scatter :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCDoubleTensor_scatterAdd"
  c_scatterAdd :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCDoubleTensor_scatterFill"
  c_scatterFill :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCDoubleTensor_gather"
  p_gather :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCDoubleTensor_scatter"
  p_scatter :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCDoubleTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCDoubleTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CInt -> Ptr CTHCudaLongTensor -> CDouble -> IO ())