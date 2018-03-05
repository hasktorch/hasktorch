{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Byte.Blas
  ( c_swap
  , c_scal
  , c_copy
  , c_axpy
  , c_dot
  , c_gemv
  , c_ger
  , c_gemm
  , p_swap
  , p_scal
  , p_copy
  , p_axpy
  , p_dot
  , p_gemv
  , p_ger
  , p_gemm
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h THByteBlas_swap"
  c_swap :: CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h THByteBlas_scal"
  c_scal :: CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h THByteBlas_copy"
  c_copy :: CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h THByteBlas_axpy"
  c_axpy :: CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h THByteBlas_dot"
  c_dot :: CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (CUChar)

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THByteBlas_gemv"
  c_gemv :: CChar -> CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THByteBlas_ger"
  c_ger :: CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (())

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THByteBlas_gemm"
  c_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (())

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THByteBlas_swap"
  p_swap :: FunPtr (CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THByteBlas_scal"
  p_scal :: FunPtr (CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THByteBlas_copy"
  p_copy :: FunPtr (CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THByteBlas_axpy"
  p_axpy :: FunPtr (CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THByteBlas_dot"
  p_dot :: FunPtr (CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (CUChar))

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THByteBlas_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THByteBlas_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> IO (()))

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THByteBlas_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> Ptr (CUChar) -> CLLong -> CUChar -> Ptr (CUChar) -> CLLong -> IO (()))