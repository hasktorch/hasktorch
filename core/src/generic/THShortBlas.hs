{-# LANGUAGE ForeignFunctionInterface #-}

module THShortBlas (
    c_THShortBlas_swap,
    c_THShortBlas_scal,
    c_THShortBlas_copy,
    c_THShortBlas_axpy,
    c_THShortBlas_dot,
    c_THShortBlas_gemv,
    c_THShortBlas_ger,
    c_THShortBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortBlas_swap : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THShortBlas_swap"
  c_THShortBlas_swap :: CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_scal : n a x incx -> void
foreign import ccall unsafe "THBlas.h THShortBlas_scal"
  c_THShortBlas_scal :: CLong -> CShort -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_copy : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THShortBlas_copy"
  c_THShortBlas_copy :: CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_axpy : n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h THShortBlas_axpy"
  c_THShortBlas_axpy :: CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_dot : n x incx y incy -> real
foreign import ccall unsafe "THBlas.h THShortBlas_dot"
  c_THShortBlas_dot :: CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort

-- |c_THShortBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h THShortBlas_gemv"
  c_THShortBlas_gemv :: CChar -> CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h THShortBlas_ger"
  c_THShortBlas_ger :: CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ()

-- |c_THShortBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h THShortBlas_gemm"
  c_THShortBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort -> Ptr CShort -> CLong -> IO ()