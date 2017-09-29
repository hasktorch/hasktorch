{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfBlas (
    c_THHalfBlas_swap,
    c_THHalfBlas_scal,
    c_THHalfBlas_copy,
    c_THHalfBlas_axpy,
    c_THHalfBlas_dot,
    c_THHalfBlas_gemv,
    c_THHalfBlas_ger,
    c_THHalfBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfBlas_swap : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_swap"
  c_THHalfBlas_swap :: CLong -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_scal : n a x incx -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_scal"
  c_THHalfBlas_scal :: CLong -> THHalf -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_copy : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_copy"
  c_THHalfBlas_copy :: CLong -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_axpy : n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_axpy"
  c_THHalfBlas_axpy :: CLong -> THHalf -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_dot : n x incx y incy -> real
foreign import ccall unsafe "THBlas.h THHalfBlas_dot"
  c_THHalfBlas_dot :: CLong -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> THHalf

-- |c_THHalfBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_gemv"
  c_THHalfBlas_gemv :: CChar -> CLong -> CLong -> THHalf -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> THHalf -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_ger"
  c_THHalfBlas_ger :: CLong -> CLong -> THHalf -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> IO ()

-- |c_THHalfBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h THHalfBlas_gemm"
  c_THHalfBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> THHalf -> Ptr THHalf -> CLong -> Ptr THHalf -> CLong -> THHalf -> Ptr THHalf -> CLong -> IO ()