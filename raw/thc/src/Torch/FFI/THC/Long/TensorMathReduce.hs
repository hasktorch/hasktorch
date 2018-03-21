{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_sum"
  c_sum :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_prod"
  c_prod :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_sumall"
  c_sumall :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_prodall"
  c_prodall :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_min"
  c_min :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_max"
  c_max :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_minall"
  c_minall :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_maxall"
  c_maxall :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_medianall"
  c_medianall :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaLongTensor_median"
  c_median :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_sum"
  p_sum :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_prod"
  p_prod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_min"
  p_min :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_max"
  p_max :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_minall"
  p_minall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> IO CLong)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaLongTensor_median"
  p_median :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaLongTensor -> CInt -> CInt -> IO ())