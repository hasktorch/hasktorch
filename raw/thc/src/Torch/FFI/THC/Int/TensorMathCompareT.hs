{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.TensorMathCompareT
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
foreign import ccall "THCTensorMathCompareT.h THIntTensor_ltTensor"
  c_ltTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_gtTensor"
  c_gtTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_leTensor"
  c_leTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_geTensor"
  c_geTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_eqTensor"
  c_eqTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_neTensor"
  c_neTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_leTensorT"
  c_leTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_geTensorT"
  c_geTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THIntTensor_neTensorT"
  c_neTensorT :: Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THIntTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> IO (()))