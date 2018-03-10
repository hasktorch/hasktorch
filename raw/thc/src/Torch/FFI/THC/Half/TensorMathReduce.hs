{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_sum"
  c_sum :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_prod"
  c_prod :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_sumall"
  c_sumall :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CFloat

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_prodall"
  c_prodall :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CFloat

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_min"
  c_min :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_max"
  c_max :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_minall"
  c_minall :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_maxall"
  c_maxall :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_medianall"
  c_medianall :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCHalfTensor_median"
  c_median :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_sum"
  p_sum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_prod"
  p_prod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_sumall"
  p_sumall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CFloat)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_prodall"
  p_prodall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CFloat)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_min"
  p_min :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_max"
  p_max :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_minall"
  p_minall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_maxall"
  p_maxall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_medianall"
  p_medianall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> IO CTHHalf)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCHalfTensor_median"
  p_median :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaHalfTensor -> CInt -> CInt -> IO ())