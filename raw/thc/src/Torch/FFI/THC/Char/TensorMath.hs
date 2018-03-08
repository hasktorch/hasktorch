{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.TensorMath
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
  , p_range
  , p_arange
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_fill :  state self value -> void
foreign import ccall "THCTensorMath.h THCharTensor_fill"
  c_fill :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> IO (())

-- | c_zero :  state self -> void
foreign import ccall "THCTensorMath.h THCharTensor_zero"
  c_zero :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (())

-- | c_zeros :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCharTensor_zeros"
  c_zeros :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_zerosLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCharTensor_zerosLike"
  c_zerosLike :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_ones :  state r_ size -> void
foreign import ccall "THCTensorMath.h THCharTensor_ones"
  c_ones :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_onesLike :  state r_ input -> void
foreign import ccall "THCTensorMath.h THCharTensor_onesLike"
  c_onesLike :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_reshape :  state r_ t size -> void
foreign import ccall "THCTensorMath.h THCharTensor_reshape"
  c_reshape :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (())

-- | c_numel :  state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h THCharTensor_numel"
  c_numel :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (CPtrdiff)

-- | c_cat :  state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h THCharTensor_cat"
  c_cat :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (())

-- | c_catArray :  state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h THCharTensor_catArray"
  c_catArray :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (Ptr (CTHCharTensor)) -> CInt -> CInt -> IO (())

-- | c_nonzero :  state tensor self -> void
foreign import ccall "THCTensorMath.h THCharTensor_nonzero"
  c_nonzero :: Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (())

-- | c_tril :  state self src k -> void
foreign import ccall "THCTensorMath.h THCharTensor_tril"
  c_tril :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (())

-- | c_triu :  state self src k -> void
foreign import ccall "THCTensorMath.h THCharTensor_triu"
  c_triu :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (())

-- | c_diag :  state self src k -> void
foreign import ccall "THCTensorMath.h THCharTensor_diag"
  c_diag :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (())

-- | c_eye :  state self n k -> void
foreign import ccall "THCTensorMath.h THCharTensor_eye"
  c_eye :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLLong -> CLLong -> IO (())

-- | c_trace :  state self -> accreal
foreign import ccall "THCTensorMath.h THCharTensor_trace"
  c_trace :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (CLong)

-- | c_range :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCharTensor_range"
  c_range :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (())

-- | c_arange :  state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h THCharTensor_arange"
  c_arange :: Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (())

-- | p_fill : Pointer to function : state self value -> void
foreign import ccall "THCTensorMath.h &THCharTensor_fill"
  p_fill :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CChar -> IO (()))

-- | p_zero : Pointer to function : state self -> void
foreign import ccall "THCTensorMath.h &THCharTensor_zero"
  p_zero :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_zeros : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCharTensor_zeros"
  p_zeros :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_zerosLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCharTensor_zerosLike"
  p_zerosLike :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_ones : Pointer to function : state r_ size -> void
foreign import ccall "THCTensorMath.h &THCharTensor_ones"
  p_ones :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_onesLike : Pointer to function : state r_ input -> void
foreign import ccall "THCTensorMath.h &THCharTensor_onesLike"
  p_onesLike :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_reshape : Pointer to function : state r_ t size -> void
foreign import ccall "THCTensorMath.h &THCharTensor_reshape"
  p_reshape :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_numel : Pointer to function : state t -> ptrdiff_t
foreign import ccall "THCTensorMath.h &THCharTensor_numel"
  p_numel :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (CPtrdiff))

-- | p_cat : Pointer to function : state result ta tb dimension -> void
foreign import ccall "THCTensorMath.h &THCharTensor_cat"
  p_cat :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CInt -> IO (()))

-- | p_catArray : Pointer to function : state result inputs numInputs dimension -> void
foreign import ccall "THCTensorMath.h &THCharTensor_catArray"
  p_catArray :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (Ptr (CTHCharTensor)) -> CInt -> CInt -> IO (()))

-- | p_nonzero : Pointer to function : state tensor self -> void
foreign import ccall "THCTensorMath.h &THCharTensor_nonzero"
  p_nonzero :: FunPtr (Ptr (CTHState) -> Ptr (CTHLongTensor) -> Ptr (CTHCharTensor) -> IO (()))

-- | p_tril : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCharTensor_tril"
  p_tril :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (()))

-- | p_triu : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCharTensor_triu"
  p_triu :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (()))

-- | p_diag : Pointer to function : state self src k -> void
foreign import ccall "THCTensorMath.h &THCharTensor_diag"
  p_diag :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> Ptr (CTHCharTensor) -> CLLong -> IO (()))

-- | p_eye : Pointer to function : state self n k -> void
foreign import ccall "THCTensorMath.h &THCharTensor_eye"
  p_eye :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLLong -> CLLong -> IO (()))

-- | p_trace : Pointer to function : state self -> accreal
foreign import ccall "THCTensorMath.h &THCharTensor_trace"
  p_trace :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> IO (CLong))

-- | p_range : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCharTensor_range"
  p_range :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (()))

-- | p_arange : Pointer to function : state r_ xmin xmax step -> void
foreign import ccall "THCTensorMath.h &THCharTensor_arange"
  p_arange :: FunPtr (Ptr (CTHState) -> Ptr (CTHCharTensor) -> CLong -> CLong -> CLong -> IO (()))