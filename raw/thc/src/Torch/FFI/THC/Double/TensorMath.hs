{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMath where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_fill :  state self value -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_fill"
  c_fill :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ()

-- | c_zero :  state self -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_zero"
  c_zero :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_zeros :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_zeros"
  c_zeros :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | c_zerosLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_zerosLike"
  c_zerosLike :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_ones :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_ones"
  c_ones :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | c_onesLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_onesLike"
  c_onesLike :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_reshape :  state r_ t size -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_reshape"
  c_reshape :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ()

-- | c_numel :  state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_numel"
  c_numel :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff

-- | c_cat :  state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_cat"
  c_cat :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ()

-- | c_catArray :  state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_catArray"
  c_catArray :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr (Ptr C'THCudaDoubleTensor) -> CInt -> CInt -> IO ()

-- | c_nonzero :  state tensor self -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_nonzero"
  c_nonzero :: Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ()

-- | c_tril :  state self src k -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_tril"
  c_tril :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ()

-- | c_triu :  state self src k -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_triu"
  c_triu :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ()

-- | c_diag :  state self src k -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_diag"
  c_diag :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ()

-- | c_eye :  state self n k -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_eye"
  c_eye :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO ()

-- | c_trace :  state self -> accreal
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_trace"
  c_trace :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CDouble

-- | c_linspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_linspace"
  c_linspace :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_logspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_logspace"
  c_logspace :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_range :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_range"
  c_range :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_arange :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCudaDoubleTensor_arange"
  c_arange :: Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | p_fill : Pointer to function : state self value -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_fill"
  p_fill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> IO ())

-- | p_zero : Pointer to function : state self -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_zero"
  p_zero :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_zeros : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_zeros"
  p_zeros :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_ones : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_ones"
  p_ones :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_onesLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_onesLike"
  p_onesLike :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_reshape : Pointer to function : state r_ t size -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_reshape"
  p_reshape :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THLongStorage -> IO ())

-- | p_numel : Pointer to function : state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_numel"
  p_numel :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CPtrdiff)

-- | p_cat : Pointer to function : state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_cat"
  p_cat :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_catArray"
  p_catArray :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr (Ptr C'THCudaDoubleTensor) -> CInt -> CInt -> IO ())

-- | p_nonzero : Pointer to function : state tensor self -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_nonzero"
  p_nonzero :: FunPtr (Ptr C'THCState -> Ptr C'THCudaLongTensor -> Ptr C'THCudaDoubleTensor -> IO ())

-- | p_tril : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_tril"
  p_tril :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_triu"
  p_triu :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ())

-- | p_diag : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_diag"
  p_diag :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> Ptr C'THCudaDoubleTensor -> CLLong -> IO ())

-- | p_eye : Pointer to function : state self n k -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_eye"
  p_eye :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CLLong -> CLLong -> IO ())

-- | p_trace : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_trace"
  p_trace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> IO CDouble)

-- | p_linspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_linspace"
  p_linspace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_logspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_logspace"
  p_logspace :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_range : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_range"
  p_range :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_arange : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCudaDoubleTensor_arange"
  p_arange :: FunPtr (Ptr C'THCState -> Ptr C'THCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())