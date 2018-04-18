{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_sum"
  c_sum :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_prod"
  c_prod :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_sumall"
  c_sumall :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CLong

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_prodall"
  c_prodall :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CLong

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_min"
  c_min :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_max"
  c_max :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_minall"
  c_minall :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_maxall"
  c_maxall :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_medianall"
  c_medianall :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCudaCharTensor_median"
  c_median :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_sum"
  p_sum :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_prod"
  p_prod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CLong)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CLong)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_min"
  p_min :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_max"
  p_max :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_minall"
  p_minall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> IO CChar)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCudaCharTensor_median"
  p_median :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaCharTensor -> CInt -> CInt -> IO ())