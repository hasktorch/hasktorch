{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMathBlas
  ( c_dot
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , p_dot
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THCCharTensor_dot"
  c_dot :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO CLong

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCCharTensor_addmv"
  c_addmv :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCCharTensor_addmm"
  c_addmm :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCCharTensor_addr"
  c_addr :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCCharTensor_addbmm"
  c_addbmm :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCCharTensor_baddbmm"
  c_baddbmm :: Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ()

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_dot"
  p_dot :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO CLong)

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_addmv"
  p_addmv :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_addmm"
  p_addmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_addr"
  p_addr :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_addbmm"
  p_addbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ())

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCCharTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> CChar -> Ptr CTHCudaCharTensor -> Ptr CTHCudaCharTensor -> IO ())