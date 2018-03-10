{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCCharTensor_sum"
  c_sum :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCCharTensor_prod"
  c_prod :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCCharTensor_sumall"
  c_sumall :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CLong

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCCharTensor_prodall"
  c_prodall :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CLong

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCCharTensor_min"
  c_min :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCCharTensor_max"
  c_max :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCCharTensor_minall"
  c_minall :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCCharTensor_maxall"
  c_maxall :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCCharTensor_medianall"
  c_medianall :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCCharTensor_median"
  c_median :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_sum"
  p_sum :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_prod"
  p_prod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_sumall"
  p_sumall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CLong)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_prodall"
  p_prodall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CLong)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_min"
  p_min :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_max"
  p_max :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_minall"
  p_minall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_maxall"
  p_maxall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_medianall"
  p_medianall :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> IO CChar)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCCharTensor_median"
  p_median :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaCharTensor -> CInt -> CInt -> IO ())