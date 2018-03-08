{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorMath
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
foreign import ccall "THCTensorMath.h THDoubleTensor_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (())

-- | c_zero :  state self -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_zero"
  c_zero :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_zeros :  state r_ size -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_zeros"
  c_zeros :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_zerosLike"
  c_zerosLike :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_ones :  state r_ size -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_ones"
  c_ones :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_onesLike"
  c_onesLike :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_reshape :  state r_ t size -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_reshape"
  c_reshape :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_numel :  state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h THDoubleTensor_numel"
  c_numel :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (CPtrdiff)

-- | c_cat :  state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_cat"
  c_cat :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (())

-- | c_catArray :  state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_catArray"
  c_catArray :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (Ptr (CTHDoubleTensor)) -> CInt -> CInt -> IO (())

-- | c_nonzero :  state tensor self -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_nonzero"
  c_nonzero :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_tril :  state self src k -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_tril"
  c_tril :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (())

-- | c_triu :  state self src k -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_triu"
  c_triu :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (())

-- | c_diag :  state self src k -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_diag"
  c_diag :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (())

-- | c_eye :  state self n k -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_eye"
  c_eye :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> CLLong -> IO (())

-- | c_trace :  state self -> accreal
foreign import ccall "THCTensorMath.h THDoubleTensor_trace"
  c_trace :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (CDouble)

-- | c_linspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_linspace"
  c_linspace :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CLLong -> IO (())

-- | c_logspace :  state r_ a b n -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_logspace"
  c_logspace :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CLLong -> IO (())

-- | c_range :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_range"
  c_range :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO (())

-- | c_arange :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THDoubleTensor_arange"
  c_arange :: Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO (())

-- | p_fill : Pointer to function : state self value -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> IO (()))

-- | p_zero : Pointer to function : state self -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_zero"
  p_zero :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_zeros : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_zeros"
  p_zeros :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_ones : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_ones"
  p_ones :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_reshape : Pointer to function : state r_ t size -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_reshape"
  p_reshape :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_numel : Pointer to function : state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h &THDoubleTensor_numel"
  p_numel :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (CPtrdiff))

-- | p_cat : Pointer to function : state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_cat"
  p_cat :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_catArray"
  p_catArray :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (Ptr (CTHDoubleTensor)) -> CInt -> CInt -> IO (()))

-- | p_nonzero : Pointer to function : state tensor self -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_tril : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_tril"
  p_tril :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_triu"
  p_triu :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (()))

-- | p_diag : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_diag"
  p_diag :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> CLLong -> IO (()))

-- | p_eye : Pointer to function : state self n k -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_eye"
  p_eye :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CLLong -> CLLong -> IO (()))

-- | p_trace : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMath.h &THDoubleTensor_trace"
  p_trace :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> IO (CDouble))

-- | p_linspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_linspace"
  p_linspace :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CLLong -> IO (()))

-- | p_logspace : Pointer to function : state r_ a b n -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_logspace"
  p_logspace :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CLLong -> IO (()))

-- | p_range : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_range"
  p_range :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO (()))

-- | p_arange : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THDoubleTensor_arange"
  p_arange :: FunPtr (Ptr (CTHState) -> Ptr (CTHDoubleTensor) -> CDouble -> CDouble -> CDouble -> IO (()))