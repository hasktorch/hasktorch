{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathBlas
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
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_dot"
  c_dot :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CFloat

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_addmv"
  c_addmv :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_addmm"
  c_addmm :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_addr"
  c_addr :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_addbmm"
  c_addbmm :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCHalfTensor_baddbmm"
  c_baddbmm :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_dot"
  p_dot :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO CFloat)

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_addmv"
  p_addmv :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_addmm"
  p_addmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_addr"
  p_addr :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_addbmm"
  p_addbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCHalfTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> CTHHalf -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())