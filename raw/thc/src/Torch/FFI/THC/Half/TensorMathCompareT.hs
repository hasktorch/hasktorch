{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathCompareT
  ( c_ltTensor
  , c_gtTensor
  , c_leTensor
  , c_geTensor
  , c_eqTensor
  , c_neTensor
  , c_ltTensorT
  , c_gtTensorT
  , c_leTensorT
  , c_geTensorT
  , c_eqTensorT
  , c_neTensorT
  , p_ltTensor
  , p_gtTensor
  , p_leTensor
  , p_geTensor
  , p_eqTensor
  , p_neTensor
  , p_ltTensorT
  , p_gtTensorT
  , p_leTensorT
  , p_geTensorT
  , p_eqTensorT
  , p_neTensorT
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_ltTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_ltTensor"
  c_ltTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_gtTensor"
  c_gtTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_leTensor"
  c_leTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_geTensor"
  c_geTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_eqTensor"
  c_eqTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_neTensor"
  c_neTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_ltTensorT"
  c_ltTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_gtTensorT"
  c_gtTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_leTensorT"
  c_leTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_geTensorT"
  c_geTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_eqTensorT"
  c_eqTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCHalfTensor_neTensorT"
  c_neTensorT :: Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ()

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_leTensor"
  p_leTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_geTensor"
  p_geTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_neTensor"
  p_neTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCHalfTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> Ptr CTHCudaHalfTensor -> IO ())