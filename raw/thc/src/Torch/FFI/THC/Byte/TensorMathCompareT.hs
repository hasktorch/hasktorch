{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorMathCompareT
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
foreign import ccall "THCTensorMathCompareT.h THByteTensor_ltTensor"
  c_ltTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_gtTensor"
  c_gtTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_leTensor"
  c_leTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_geTensor"
  c_geTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_eqTensor"
  c_eqTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_neTensor"
  c_neTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_leTensorT"
  c_leTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_geTensorT"
  c_geTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THByteTensor_neTensorT"
  c_neTensorT :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (())

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THByteTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (()))