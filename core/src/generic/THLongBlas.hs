{-# LANGUAGE ForeignFunctionInterface#-}

module THLongBlas (
    c_THLongBlas_swap,
    c_THLongBlas_scal,
    c_THLongBlas_copy,
    c_THLongBlas_axpy,
    c_THLongBlas_dot,
    c_THLongBlas_gemv,
    c_THLongBlas_ger,
    c_THLongBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_swap"
  c_THLongBlas_swap :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THLongBlas_scal"
  c_THLongBlas_scal :: CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_copy"
  c_THLongBlas_copy :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_axpy"
  c_THLongBlas_axpy :: CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THLongBlas_dot"
  c_THLongBlas_dot :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong

-- |c_THLongBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THLongBlas_gemv"
  c_THLongBlas_gemv :: CChar -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THLongBlas_ger"
  c_THLongBlas_ger :: CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THLongBlas_gemm"
  c_THLongBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ()