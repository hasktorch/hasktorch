{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Half.TensorMathPairwise where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_add :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_add"
  c_add :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_sub :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_sub"
  c_sub :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_add_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_add_scaled"
  c_add_scaled :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ()

-- | c_sub_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_sub_scaled"
  c_sub_scaled :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ()

-- | c_mul :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_mul"
  c_mul :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_div :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_div"
  c_div :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_lshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_lshift"
  c_lshift :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_rshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_rshift"
  c_rshift :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_fmod :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_fmod"
  c_fmod :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_remainder :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_remainder"
  c_remainder :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_bitand :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_bitand"
  c_bitand :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_bitor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_bitor"
  c_bitor :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_bitxor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_bitxor"
  c_bitxor :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ()

-- | c_equal :  state self src -> int
foreign import ccall "THCTensorMathPairwise.h THCudaHalfTensor_equal"
  c_equal :: Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO CInt

-- | p_add : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_add"
  p_add :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_sub : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_sub"
  p_sub :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_add_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ())

-- | p_sub_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> CTHHalf -> IO ())

-- | p_mul : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_mul"
  p_mul :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_div : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_div"
  p_div :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_lshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_lshift"
  p_lshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_rshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_rshift"
  p_rshift :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_fmod : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_fmod"
  p_fmod :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_remainder : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_remainder"
  p_remainder :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_bitand : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_bitand"
  p_bitand :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_bitor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_bitor"
  p_bitor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_bitxor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_bitxor"
  p_bitxor :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> CTHHalf -> IO ())

-- | p_equal : Pointer to function : state self src -> int
foreign import ccall "THCTensorMathPairwise.h &THCudaHalfTensor_equal"
  p_equal :: FunPtr (Ptr C'THCState -> Ptr C'THCudaHalfTensor -> Ptr C'THCudaHalfTensor -> IO CInt)