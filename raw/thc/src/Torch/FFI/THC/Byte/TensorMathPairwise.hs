{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.TensorMathPairwise
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
foreign import ccall "THCTensorMathPairwise.h THByteTensor_add"
  c_add :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_sub :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_sub"
  c_sub :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_add_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_add_scaled"
  c_add_scaled :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (())

-- | c_sub_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_sub_scaled"
  c_sub_scaled :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (())

-- | c_mul :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_mul"
  c_mul :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_div :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_div"
  c_div :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_lshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_lshift"
  c_lshift :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_rshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_rshift"
  c_rshift :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_fmod :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_fmod"
  c_fmod :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_remainder :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_remainder"
  c_remainder :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_bitand :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_bitand"
  c_bitand :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_bitor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_bitor"
  c_bitor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_bitxor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THByteTensor_bitxor"
  c_bitxor :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (())

-- | c_equal :  state self src -> int
foreign import ccall "THCTensorMathPairwise.h THByteTensor_equal"
  c_equal :: Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CInt)

-- | p_add : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_add"
  p_add :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_sub : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_sub"
  p_sub :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_add_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (()))

-- | p_sub_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> CUChar -> IO (()))

-- | p_mul : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_mul"
  p_mul :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_div : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_div"
  p_div :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_lshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_lshift"
  p_lshift :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_rshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_rshift"
  p_rshift :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_fmod : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_fmod"
  p_fmod :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_remainder : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_remainder"
  p_remainder :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_bitand : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_bitand"
  p_bitand :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_bitor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_bitor"
  p_bitor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_bitxor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_bitxor"
  p_bitxor :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> CUChar -> IO (()))

-- | p_equal : Pointer to function : state self src -> int
foreign import ccall "THCTensorMathPairwise.h &THByteTensor_equal"
  p_equal :: FunPtr (Ptr (CTHState) -> Ptr (CTHByteTensor) -> Ptr (CTHByteTensor) -> IO (CInt))