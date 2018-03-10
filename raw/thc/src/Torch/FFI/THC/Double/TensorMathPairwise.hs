{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMathPairwise
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
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_add"
  c_add :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_sub :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_sub"
  c_sub :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_add_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_add_scaled"
  c_add_scaled :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_sub_scaled :  state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_sub_scaled"
  c_sub_scaled :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> IO ()

-- | c_mul :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_mul"
  c_mul :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_div :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_div"
  c_div :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_lshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_lshift"
  c_lshift :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_rshift :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_rshift"
  c_rshift :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_fmod :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_fmod"
  c_fmod :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_remainder :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_remainder"
  c_remainder :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_bitand :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_bitand"
  c_bitand :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_bitor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_bitor"
  c_bitor :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_bitxor :  state self src value -> void
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_bitxor"
  c_bitxor :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_equal :  state self src -> int
foreign import ccall "THCTensorMathPairwise.h THCDoubleTensor_equal"
  c_equal :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO CInt

-- | p_add : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_add"
  p_add :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_sub : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_sub"
  p_sub :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_add_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_add_scaled"
  p_add_scaled :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_sub_scaled : Pointer to function : state self src value alpha -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_sub_scaled"
  p_sub_scaled :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> IO ())

-- | p_mul : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_mul"
  p_mul :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_div : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_div"
  p_div :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_lshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_lshift"
  p_lshift :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_rshift : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_rshift"
  p_rshift :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_fmod : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_fmod"
  p_fmod :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_remainder : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_remainder"
  p_remainder :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_bitand : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_bitand"
  p_bitand :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_bitor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_bitor"
  p_bitor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_bitxor : Pointer to function : state self src value -> void
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_bitxor"
  p_bitxor :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_equal : Pointer to function : state self src -> int
foreign import ccall "THCTensorMathPairwise.h &THCDoubleTensor_equal"
  p_equal :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO CInt)