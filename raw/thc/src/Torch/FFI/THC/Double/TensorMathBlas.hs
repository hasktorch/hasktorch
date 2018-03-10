{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMathBlas
  ( c_dot
  , c_addmv
  , c_addmm
  , c_addr
  , c_addbmm
  , c_baddbmm
  , c_btrifact
  , c_btrisolve
  , p_dot
  , p_addmv
  , p_addmm
  , p_addr
  , p_addbmm
  , p_baddbmm
  , p_btrifact
  , p_btrisolve
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_dot :  state self src -> accreal
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_dot"
  c_dot :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO CDouble

-- | c_addmv :  state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_addmv"
  c_addmv :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_addmm :  state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_addmm"
  c_addmm :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_addr :  state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_addr"
  c_addr :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_addbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_addbmm"
  c_addbmm :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_baddbmm :  state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_baddbmm"
  c_baddbmm :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_btrifact :  state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_btrifact"
  c_btrifact :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_btrisolve :  state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h THCDoubleTensor_btrisolve"
  c_btrisolve :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ()

-- | p_dot : Pointer to function : state self src -> accreal
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_dot"
  p_dot :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO CDouble)

-- | p_addmv : Pointer to function : state self beta t alpha mat vec -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_addmv"
  p_addmv :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_addmm : Pointer to function : state self beta t alpha mat1 mat2 -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_addmm"
  p_addmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_addr : Pointer to function : state self beta t alpha vec1 vec2 -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_addr"
  p_addr :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_addbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_addbmm"
  p_addbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_baddbmm : Pointer to function : state result beta t alpha batch1 batch2 -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_baddbmm"
  p_baddbmm :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> CDouble -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_btrifact : Pointer to function : state ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_btrifact"
  p_btrifact :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> Ptr CTHCudaIntTensor -> CInt -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_btrisolve : Pointer to function : state rb_ b atf pivots -> void
foreign import ccall "THCTensorMathBlas.h &THCDoubleTensor_btrisolve"
  p_btrisolve :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaIntTensor -> IO ())