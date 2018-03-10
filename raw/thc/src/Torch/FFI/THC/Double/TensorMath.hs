{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Double.TensorMath
  ( c_fill
  , c_zero
  , c_zeros
  , c_zerosLike
  , c_ones
  , c_onesLike
  , c_reshape
  , c_numel
  , c_cat
  , c_catArray
  , c_nonzero
  , c_tril
  , c_triu
  , c_diag
  , c_eye
  , c_trace
  , c_linspace
  , c_logspace
  , c_range
  , c_arange
  , p_fill
  , p_zero
  , p_zeros
  , p_zerosLike
  , p_ones
  , p_onesLike
  , p_reshape
  , p_numel
  , p_cat
  , p_catArray
  , p_nonzero
  , p_tril
  , p_triu
  , p_diag
  , p_eye
  , p_trace
  , p_linspace
  , p_logspace
  , p_range
  , p_arange
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  state self value -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_fill"
  c_fill :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ()

-- | c_zero :  state self -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_zero"
  c_zero :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_zeros :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_zeros"
  c_zeros :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_zerosLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_zerosLike"
  c_zerosLike :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_ones :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_ones"
  c_ones :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_onesLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_onesLike"
  c_onesLike :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_reshape :  state r_ t size -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_reshape"
  c_reshape :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ()

-- | c_numel :  state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h THCDoubleTensor_numel"
  c_numel :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO CPtrdiff

-- | c_cat :  state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_cat"
  c_cat :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> IO ()

-- | c_catArray :  state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_catArray"
  c_catArray :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr (Ptr CTHCudaDoubleTensor) -> CInt -> CInt -> IO ()

-- | c_nonzero :  state tensor self -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_nonzero"
  c_nonzero :: Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ()

-- | c_tril :  state self src k -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_tril"
  c_tril :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ()

-- | c_triu :  state self src k -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_triu"
  c_triu :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ()

-- | c_diag :  state self src k -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_diag"
  c_diag :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ()

-- | c_eye :  state self n k -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_eye"
  c_eye :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CLLong -> CLLong -> IO ()

-- | c_trace :  state self -> accreal
foreign import ccall "THCTensorMath.h THCDoubleTensor_trace"
  c_trace :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO CDouble

-- | c_linspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_linspace"
  c_linspace :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_logspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_logspace"
  c_logspace :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ()

-- | c_range :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_range"
  c_range :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | c_arange :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCDoubleTensor_arange"
  c_arange :: Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ()

-- | p_fill : Pointer to function : state self value -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_fill"
  p_fill :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> IO ())

-- | p_zero : Pointer to function : state self -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_zero"
  p_zero :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_zeros : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_zeros"
  p_zeros :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_zerosLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_ones : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_ones"
  p_ones :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_onesLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_onesLike"
  p_onesLike :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_reshape : Pointer to function : state r_ t size -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_reshape"
  p_reshape :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaLongStorage -> IO ())

-- | p_numel : Pointer to function : state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h &THCDoubleTensor_numel"
  p_numel :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO CPtrdiff)

-- | p_cat : Pointer to function : state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_cat"
  p_cat :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CInt -> IO ())

-- | p_catArray : Pointer to function : state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_catArray"
  p_catArray :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr (Ptr CTHCudaDoubleTensor) -> CInt -> CInt -> IO ())

-- | p_nonzero : Pointer to function : state tensor self -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_nonzero"
  p_nonzero :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaLongTensor -> Ptr CTHCudaDoubleTensor -> IO ())

-- | p_tril : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_tril"
  p_tril :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ())

-- | p_triu : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_triu"
  p_triu :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ())

-- | p_diag : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_diag"
  p_diag :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> Ptr CTHCudaDoubleTensor -> CLLong -> IO ())

-- | p_eye : Pointer to function : state self n k -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_eye"
  p_eye :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CLLong -> CLLong -> IO ())

-- | p_trace : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMath.h &THCDoubleTensor_trace"
  p_trace :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> IO CDouble)

-- | p_linspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_linspace"
  p_linspace :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_logspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_logspace"
  p_logspace :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CLLong -> IO ())

-- | p_range : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_range"
  p_range :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())

-- | p_arange : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCDoubleTensor_arange"
  p_arange :: FunPtr (Ptr CTHCudaState -> Ptr CTHCudaDoubleTensor -> CDouble -> CDouble -> CDouble -> IO ())