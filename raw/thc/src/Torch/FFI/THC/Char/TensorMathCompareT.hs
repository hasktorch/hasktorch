{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Char.TensorMathCompareT where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_ltTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_ltTensor"
  c_ltTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_gtTensor"
  c_gtTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_leTensor"
  c_leTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_geTensor"
  c_geTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_eqTensor"
  c_eqTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_neTensor"
  c_neTensor :: Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_ltTensorT"
  c_ltTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_gtTensorT"
  c_gtTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_leTensorT"
  c_leTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_geTensorT"
  c_geTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_eqTensorT"
  c_eqTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCudaCharTensor_neTensorT"
  c_neTensorT :: Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ()

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_leTensor"
  p_leTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_geTensor"
  p_geTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_neTensor"
  p_neTensor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaByteTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCudaCharTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr C'THCState -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> Ptr C'THCudaCharTensor -> IO ())