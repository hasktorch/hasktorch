{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_sum"
  c_sum :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_prod"
  c_prod :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_sumall"
  c_sumall :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CFloat

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_prodall"
  c_prodall :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CFloat

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_min"
  c_min :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_max"
  c_max :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_minall"
  c_minall :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_maxall"
  c_maxall :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_medianall"
  c_medianall :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaHalfTensor_median"
  c_median :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_sum"
  p_sum :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_prod"
  p_prod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CFloat)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CFloat)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_min"
  p_min :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_max"
  p_max :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_minall"
  p_minall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> IO CTHHalf)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaHalfTensor_median"
  p_median :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaHalfTensor -> CInt -> CInt -> IO ())