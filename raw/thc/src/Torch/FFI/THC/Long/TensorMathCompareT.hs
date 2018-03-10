{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Long.TensorMathCompareT
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
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_ltTensor"
  c_ltTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_gtTensor"
  c_gtTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_leTensor"
  c_leTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_geTensor"
  c_geTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_eqTensor"
  c_eqTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_neTensor"
  c_neTensor :: Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_ltTensorT"
  c_ltTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_gtTensorT"
  c_gtTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_leTensorT"
  c_leTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_geTensorT"
  c_geTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_eqTensorT"
  c_eqTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCLongTensor_neTensorT"
  c_neTensorT :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ()

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_leTensor"
  p_leTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_geTensor"
  p_geTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_neTensor"
  p_neTensor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaByteTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCLongTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> Ptr CTHCudaLongTensor -> IO ())