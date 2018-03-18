{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMathBlas where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THCIntTensor_dot"
  c_dot :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CLong

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCIntTensor_addmv"
  c_addmv :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCIntTensor_addmm"
  c_addmm :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCIntTensor_addr"
  c_addr :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCIntTensor_addbmm"
  c_addbmm :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCIntTensor_baddbmm"
  c_baddbmm :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_dot"
  p_dot :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO CLong)

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_addmv"
  p_addmv :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_addmm"
  p_addmm :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_addr"
  p_addr :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_addbmm"
  p_addbmm :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCIntTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> CInt -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> IO ())