{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMathPairwise
  ( c_add
  , c_sub
  , c_add_scaled
  , c_sub_scaled
  , c_mul
  , c_div
  , c_lshift
  , c_rshift
  , c_fmod
  , c_remainder
  , c_bitand
  , c_bitor
  , c_bitxor
  , c_equal
  , p_add
  , p_sub
  , p_add_scaled
  , p_sub_scaled
  , p_mul
  , p_div
  , p_lshift
  , p_rshift
  , p_fmod
  , p_remainder
  , p_bitand
  , p_bitor
  , p_bitxor
  , p_equal
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_add :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_add"
  c_add :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_sub :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_sub"
  c_sub :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_add_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_add_scaled"
  c_add_scaled :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (())

-- | c_sub_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_sub_scaled"
  c_sub_scaled :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (())

-- | c_mul :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_mul"
  c_mul :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_div :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_div"
  c_div :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_lshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_lshift"
  c_lshift :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_rshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_rshift"
  c_rshift :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_fmod :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_fmod"
  c_fmod :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_remainder :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_remainder"
  c_remainder :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_bitand :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_bitand"
  c_bitand :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_bitor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_bitor"
  c_bitor :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_bitxor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCharTensor_bitxor"
  c_bitxor :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_equal :  state self src -> int
foreign import ccall "THCTensorMathPairwise.h THCharTensor_equal"
  c_equal :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CInt)

-- | p_add : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_add"
  p_add :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_sub : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_sub"
  p_sub :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_add_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (()))

-- | p_sub_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> CChar -> IO (()))

-- | p_mul : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_mul"
  p_mul :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_div : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_div"
  p_div :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_lshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_lshift"
  p_lshift :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_rshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_rshift"
  p_rshift :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_fmod : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_fmod"
  p_fmod :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_remainder : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_remainder"
  p_remainder :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_bitand : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_bitand"
  p_bitand :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_bitor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_bitor"
  p_bitor :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_bitxor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_equal : Pointer to function : state self src -> int
foreign import ccall "THCTensorMathPairwise.h &THCharTensor_equal"
  p_equal :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (CInt))