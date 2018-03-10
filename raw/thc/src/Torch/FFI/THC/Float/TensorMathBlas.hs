{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathBlas where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_dot"
  c_dot :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO CDouble

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_addmv"
  c_addmv :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_addmm"
  c_addmm :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_addr"
  c_addr :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_addbmm"
  c_addbmm :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_baddbmm"
  c_baddbmm :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_btrifact :  state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_btrifact"
  c_btrifact :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaFloatTensor -> IO ()

-- | c_btrisolve :  state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h THCFloatTensor_btrisolve"
  c_btrisolve :: Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_dot"
  p_dot :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO CDouble)

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_addmv"
  p_addmv :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_addmm"
  p_addmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_addr"
  p_addr :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_addbmm"
  p_addbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> CFloat -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_btrifact : Pointer to function : state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_btrifact"
  p_btrifact :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaFloatTensor -> IO ())

-- | p_btrisolve : Pointer to function : state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h &THCFloatTensor_btrisolve"
  p_btrisolve :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaFloatTensor -> Ptr CTHCudaIntTensor -> IO ())