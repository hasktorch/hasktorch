{-# LANGUAGE ForeignFunctionInterface #-}

module THIntBlas (
    c_THIntBlas_swap,
    c_THIntBlas_scal,
    c_THIntBlas_copy,
    c_THIntBlas_axpy,
    c_THIntBlas_dot,
    c_THIntBlas_gemv,
    c_THIntBlas_ger,
    c_THIntBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_swap"
  c_THIntBlas_swap :: CLong -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THIntBlas_scal"
  c_THIntBlas_scal :: CLong -> CInt -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_copy"
  c_THIntBlas_copy :: CLong -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_axpy"
  c_THIntBlas_axpy :: CLong -> CInt -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THIntBlas_dot"
  c_THIntBlas_dot :: CLong -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> CInt

-- |c_THIntBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THIntBlas_gemv"
  c_THIntBlas_gemv :: CChar -> CLong -> CLong -> CInt -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> CInt -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THIntBlas_ger"
  c_THIntBlas_ger :: CLong -> CLong -> CInt -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> IO ()

-- |c_THIntBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THIntBlas_gemm"
  c_THIntBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CInt -> Ptr CInt -> CLong -> Ptr CInt -> CLong -> CInt -> Ptr CInt -> CLong -> IO ()