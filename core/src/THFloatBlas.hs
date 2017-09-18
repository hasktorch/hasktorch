{-# LANGUAGE ForeignFunctionInterface#-}

module THFloatBlas (
    c_THFloatBlas_swap,
    c_THFloatBlas_scal,
    c_THFloatBlas_copy,
    c_THFloatBlas_axpy,
    c_THFloatBlas_dot,
    c_THFloatBlas_gemv,
    c_THFloatBlas_ger,
    c_THFloatBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_swap"
  c_THFloatBlas_swap :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THFloatBlas_scal"
  c_THFloatBlas_scal :: CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_copy"
  c_THFloatBlas_copy :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_axpy"
  c_THFloatBlas_axpy :: CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THFloatBlas_dot"
  c_THFloatBlas_dot :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat

-- |c_THFloatBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THFloatBlas_gemv"
  c_THFloatBlas_gemv :: CChar -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THFloatBlas_ger"
  c_THFloatBlas_ger :: CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THFloatBlas_gemm"
  c_THFloatBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()