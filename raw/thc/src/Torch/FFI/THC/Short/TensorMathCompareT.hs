{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.TensorMathCompareT
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
foreign import ccall "THCTensorMathCompareT.h THShortTensor_ltTensor"
  c_ltTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_gtTensor"
  c_gtTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_leTensor"
  c_leTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_geTensor"
  c_geTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_eqTensor"
  c_eqTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_neTensor"
  c_neTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_leTensorT"
  c_leTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_geTensorT"
  c_geTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THShortTensor_neTensorT"
  c_neTensorT :: Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (())

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THShortTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> Ptr (CTHShortTensor) -> IO (()))