{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.TensorConv where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_THCudaTensor_conv2Dmv :  state output beta input kernel srow scol type -> void
foreign import ccall "THCTensorConv.h THCudaTensor_conv2Dmv"
  c_THCudaTensor_conv2Dmv :: Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CChar -> IO ()

-- | c_THCudaTensor_conv2Dmm :  state output beta input kernel srow scol type -> void
foreign import ccall "THCTensorConv.h THCudaTensor_conv2Dmm"
  c_THCudaTensor_conv2Dmm :: Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CChar -> IO ()

-- | c_THCudaTensor_conv2DRevger :  state output beta alpha input kernel srow scol -> void
foreign import ccall "THCTensorConv.h THCudaTensor_conv2DRevger"
  c_THCudaTensor_conv2DRevger :: Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> IO ()

-- | c_THCudaTensor_conv2DRevgerm :  state output beta alpha input kernel srow scol -> void
foreign import ccall "THCTensorConv.h THCudaTensor_conv2DRevgerm"
  c_THCudaTensor_conv2DRevgerm :: Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> IO ()

-- | c_THCudaTensor_conv2Dmap :  state output input kernel stride_x stride_y table fanin -> void
foreign import ccall "THCTensorConv.h THCudaTensor_conv2Dmap"
  c_THCudaTensor_conv2Dmap :: Ptr CTHCState -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CTHCudaTensor -> CLLong -> IO ()

-- | p_THCudaTensor_conv2Dmv : Pointer to function : state output beta input kernel srow scol type -> void
foreign import ccall "THCTensorConv.h &THCudaTensor_conv2Dmv"
  p_THCudaTensor_conv2Dmv :: FunPtr (Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CChar -> IO ())

-- | p_THCudaTensor_conv2Dmm : Pointer to function : state output beta input kernel srow scol type -> void
foreign import ccall "THCTensorConv.h &THCudaTensor_conv2Dmm"
  p_THCudaTensor_conv2Dmm :: FunPtr (Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CChar -> IO ())

-- | p_THCudaTensor_conv2DRevger : Pointer to function : state output beta alpha input kernel srow scol -> void
foreign import ccall "THCTensorConv.h &THCudaTensor_conv2DRevger"
  p_THCudaTensor_conv2DRevger :: FunPtr (Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> IO ())

-- | p_THCudaTensor_conv2DRevgerm : Pointer to function : state output beta alpha input kernel srow scol -> void
foreign import ccall "THCTensorConv.h &THCudaTensor_conv2DRevgerm"
  p_THCudaTensor_conv2DRevgerm :: FunPtr (Ptr CTHCState -> Ptr CTHCudaTensor -> CFloat -> CFloat -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> IO ())

-- | p_THCudaTensor_conv2Dmap : Pointer to function : state output input kernel stride_x stride_y table fanin -> void
foreign import ccall "THCTensorConv.h &THCudaTensor_conv2Dmap"
  p_THCudaTensor_conv2Dmap :: FunPtr (Ptr CTHCState -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> Ptr CTHCudaTensor -> CLLong -> CLLong -> Ptr CTHCudaTensor -> CLLong -> IO ())