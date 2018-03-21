{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMathCompareT where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_ltTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_ltTensor"
  c_ltTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_gtTensor"
  c_gtTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_leTensor"
  c_leTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_geTensor"
  c_geTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_eqTensor"
  c_eqTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_neTensor"
  c_neTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_ltTensorT"
  c_ltTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_gtTensorT"
  c_gtTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_leTensorT"
  c_leTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_geTensorT"
  c_geTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_eqTensorT"
  c_eqTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaFloatTensor_neTensorT"
  c_neTensorT :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_leTensor"
  p_leTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_geTensor"
  p_geTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_neTensor"
  p_neTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaFloatTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())