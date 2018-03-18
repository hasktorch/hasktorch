{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Float.TensorMath where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_fill :  state self value -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> IO ()

-- | c_zero :  state self -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_zero"
  c_zero :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_zeros :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_zeros"
  c_zeros :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ()

-- | c_zerosLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_zerosLike"
  c_zerosLike :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_ones :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_ones"
  c_ones :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ()

-- | c_onesLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_onesLike"
  c_onesLike :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_reshape :  state r_ t size -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_reshape"
  c_reshape :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ()

-- | c_numel :  state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h THCFloatTensor_numel"
  c_numel :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO CPtrdiff

-- | c_cat :  state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_cat"
  c_cat :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ()

-- | c_catArray :  state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_catArray"
  c_catArray :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr (Ptr C'THCudaFloatTensor) -> CInt -> CInt -> IO ()

-- | c_nonzero :  state tensor self -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_nonzero"
  c_nonzero :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ()

-- | c_tril :  state self src k -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_tril"
  c_tril :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ()

-- | c_triu :  state self src k -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_triu"
  c_triu :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ()

-- | c_diag :  state self src k -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_diag"
  c_diag :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ()

-- | c_eye :  state self n k -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_eye"
  c_eye :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CLLong -> CLLong -> IO ()

-- | c_trace :  state self -> accreal
foreign import ccall "THCTensorMath.h THCFloatTensor_trace"
  c_trace :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO CDouble

-- | c_linspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_linspace"
  c_linspace :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> CLLong -> IO ()

-- | c_logspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_logspace"
  c_logspace :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> CLLong -> IO ()

-- | c_range :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_range"
  c_range :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_arange :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCFloatTensor_arange"
  c_arange :: Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | p_fill : Pointer to function : state self value -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> IO ())

-- | p_zero : Pointer to function : state self -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_zero"
  p_zero :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_zeros : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_zeros"
  p_zeros :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_ones : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_ones"
  p_ones :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ())

-- | p_onesLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_onesLike"
  p_onesLike :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_reshape : Pointer to function : state r_ t size -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_reshape"
  p_reshape :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCLongStorage -> IO ())

-- | p_numel : Pointer to function : state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h &THCFloatTensor_numel"
  p_numel :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO CPtrdiff)

-- | p_cat : Pointer to function : state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_cat"
  p_cat :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_catArray"
  p_catArray :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr (Ptr C'THCudaFloatTensor) -> CInt -> CInt -> IO ())

-- | p_nonzero : Pointer to function : state tensor self -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_nonzero"
  p_nonzero :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaFloatTensor -> IO ())

-- | p_tril : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_tril"
  p_tril :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_triu"
  p_triu :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ())

-- | p_diag : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_diag"
  p_diag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> Ptr C'THCudaFloatTensor -> CLLong -> IO ())

-- | p_eye : Pointer to function : state self n k -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_eye"
  p_eye :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CLLong -> CLLong -> IO ())

-- | p_trace : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMath.h &THCFloatTensor_trace"
  p_trace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> IO CDouble)

-- | p_linspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_linspace"
  p_linspace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> CLLong -> IO ())

-- | p_logspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_logspace"
  p_logspace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CFloat -> CFloat -> CLLong -> IO ())

-- | p_range : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_range"
  p_range :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_arange : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCFloatTensor_arange"
  p_arange :: FunPtr (Ptr C'THCState -> Ptr C'THCudaFloatTensor -> CDouble -> CDouble -> CDouble -> IO ())