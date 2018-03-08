{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMathCompareT
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
foreign import ccall "THCTensorMathCompareT.h THCharTensor_ltTensor"
  c_ltTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_gtTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_gtTensor"
  c_gtTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_leTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_leTensor"
  c_leTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_geTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_geTensor"
  c_geTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_eqTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_eqTensor"
  c_eqTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_neTensor :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_neTensor"
  c_neTensor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_ltTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_ltTensorT"
  c_ltTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_gtTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_gtTensorT"
  c_gtTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_leTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_leTensorT"
  c_leTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_geTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_geTensorT"
  c_geTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_eqTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_eqTensorT"
  c_eqTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_neTensorT :  state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h THCharTensor_neTensorT"
  c_neTensorT :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | p_ltTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_ltTensor"
  p_ltTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_gtTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_gtTensor"
  p_gtTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_leTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_leTensor"
  p_leTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_geTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_geTensor"
  p_geTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_eqTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_eqTensor"
  p_eqTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_neTensor : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_neTensor"
  p_neTensor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_ltTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_ltTensorT"
  p_ltTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_gtTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_gtTensorT"
  p_gtTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_leTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_leTensorT"
  p_leTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_geTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_geTensorT"
  p_geTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_eqTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_eqTensorT"
  p_eqTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_neTensorT : Pointer to function : state self_ src1 src2 -> void
foreign import ccall "THCTensorMathCompareT.h &THCharTensor_neTensorT"
  p_neTensorT :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))