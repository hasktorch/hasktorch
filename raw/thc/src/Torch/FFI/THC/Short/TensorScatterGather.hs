{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorScatterGather where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gather :  state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h THCShortTensor_gather"
  c_gather :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ()

-- | c_scatter :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCShortTensor_scatter"
  c_scatter :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> IO ()

-- | c_scatterAdd :  state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h THCShortTensor_scatterAdd"
  c_scatterAdd :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> IO ()

-- | c_scatterFill :  state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h THCShortTensor_scatterFill"
  c_scatterFill :: Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> CShort -> IO ()

-- | p_gather : Pointer to function : state tensor src dim index -> void
foreign import ccall "THCTensorScatterGather.h &THCShortTensor_gather"
  p_gather :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> IO ())

-- | p_scatter : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCShortTensor_scatter"
  p_scatter :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> IO ())

-- | p_scatterAdd : Pointer to function : state tensor dim index src -> void
foreign import ccall "THCTensorScatterGather.h &THCShortTensor_scatterAdd"
  p_scatterAdd :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> Ptr CTHCudaShortTensor -> IO ())

-- | p_scatterFill : Pointer to function : state tensor dim index value -> void
foreign import ccall "THCTensorScatterGather.h &THCShortTensor_scatterFill"
  p_scatterFill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaShortTensor -> CInt -> Ptr CTHCudaLongTensor -> CShort -> IO ())