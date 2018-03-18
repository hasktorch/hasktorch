{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Short.TensorMathReduce where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_sum :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCShortTensor_sum"
  c_sum :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | c_prod :  state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCShortTensor_prod"
  c_prod :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | c_sumall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCShortTensor_sumall"
  c_sumall :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CLong

-- | c_prodall :  state self -> accreal
foreign import ccall "THCTensorMathReduce.h THCShortTensor_prodall"
  c_prodall :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CLong

-- | c_min :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCShortTensor_min"
  c_min :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | c_max :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCShortTensor_max"
  c_max :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | c_minall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCShortTensor_minall"
  c_minall :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort

-- | c_maxall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCShortTensor_maxall"
  c_maxall :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort

-- | c_medianall :  state self -> real
foreign import ccall "THCTensorMathReduce.h THCShortTensor_medianall"
  c_medianall :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort

-- | c_median :  state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h THCShortTensor_median"
  c_median :: Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ()

-- | p_sum : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_sum"
  p_sum :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())

-- | p_prod : Pointer to function : state self src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_prod"
  p_prod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())

-- | p_sumall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_sumall"
  p_sumall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CLong)

-- | p_prodall : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_prodall"
  p_prodall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CLong)

-- | p_min : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_min"
  p_min :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())

-- | p_max : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_max"
  p_max :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())

-- | p_minall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_minall"
  p_minall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort)

-- | p_maxall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_maxall"
  p_maxall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort)

-- | p_medianall : Pointer to function : state self -> real
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_medianall"
  p_medianall :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> IO CShort)

-- | p_median : Pointer to function : state values indices src dim keepdim -> void
foreign import ccall "THCTensorMathReduce.h &THCShortTensor_median"
  p_median :: FunPtr (Ptr C'THCState -> Ptr C'THCudaShortTensor -> Ptr C'THCudaLongTensor -> Ptr C'THCudaShortTensor -> CInt -> CInt -> IO ())