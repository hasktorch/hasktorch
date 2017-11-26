{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfBlas (
    c_THHalfBlas_swap,
    c_THHalfBlas_scal,
    c_THHalfBlas_copy,
    c_THHalfBlas_axpy,
    c_THHalfBlas_dot,
    c_THHalfBlas_gemv,
    c_THHalfBlas_ger,
    c_THHalfBlas_gemm,
    p_THHalfBlas_swap,
    p_THHalfBlas_scal,
    p_THHalfBlas_copy,
    p_THHalfBlas_axpy,
    p_THHalfBlas_dot,
    p_THHalfBlas_gemv,
    p_THHalfBlas_ger,
    p_THHalfBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THHalfBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_swap"
  c_THHalfBlas_swap :: CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THHalfBlas_scal"
  c_THHalfBlas_scal :: CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_copy"
  c_THHalfBlas_copy :: CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_axpy"
  c_THHalfBlas_axpy :: CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THHalfBlas_dot"
  c_THHalfBlas_dot :: CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf

-- |c_THHalfBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THHalfBlas_gemv"
  c_THHalfBlas_gemv :: CChar -> CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THHalfBlas_ger"
  c_THHalfBlas_ger :: CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ()

-- |c_THHalfBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THHalfBlas_gemm"
  c_THHalfBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ()

-- |p_THHalfBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_swap"
  p_THHalfBlas_swap :: FunPtr (CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THHalfBlas_scal"
  p_THHalfBlas_scal :: FunPtr (CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_copy"
  p_THHalfBlas_copy :: FunPtr (CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_axpy"
  p_THHalfBlas_axpy :: FunPtr (CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THHalfBlas_dot"
  p_THHalfBlas_dot :: FunPtr (CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf)

-- |p_THHalfBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_gemv"
  p_THHalfBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THHalfBlas_ger"
  p_THHalfBlas_ger :: FunPtr (CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> IO ())

-- |p_THHalfBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THHalfBlas_gemm"
  p_THHalfBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> Ptr THHalf -> CLLong -> THHalf -> Ptr THHalf -> CLLong -> IO ())