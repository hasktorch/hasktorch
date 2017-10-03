{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleBlas (
    c_THDoubleBlas_swap,
    c_THDoubleBlas_scal,
    c_THDoubleBlas_copy,
    c_THDoubleBlas_axpy,
    c_THDoubleBlas_dot,
    c_THDoubleBlas_gemv,
    c_THDoubleBlas_ger,
    c_THDoubleBlas_gemm,
    p_THDoubleBlas_swap,
    p_THDoubleBlas_scal,
    p_THDoubleBlas_copy,
    p_THDoubleBlas_axpy,
    p_THDoubleBlas_dot,
    p_THDoubleBlas_gemv,
    p_THDoubleBlas_ger,
    p_THDoubleBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleBlas_swap : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_swap"
  c_THDoubleBlas_swap :: CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_scal : n a x incx -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_scal"
  c_THDoubleBlas_scal :: CLong -> CDouble -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_copy : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_copy"
  c_THDoubleBlas_copy :: CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_axpy : n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_axpy"
  c_THDoubleBlas_axpy :: CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_dot : n x incx y incy -> real
foreign import ccall unsafe "THBlas.h THDoubleBlas_dot"
  c_THDoubleBlas_dot :: CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble

-- |c_THDoubleBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_gemv"
  c_THDoubleBlas_gemv :: CChar -> CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_ger"
  c_THDoubleBlas_ger :: CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ()

-- |c_THDoubleBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h THDoubleBlas_gemm"
  c_THDoubleBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble -> Ptr CDouble -> CLong -> IO ()

-- |p_THDoubleBlas_swap : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_swap"
  p_THDoubleBlas_swap :: FunPtr (CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_scal : Pointer to function n a x incx -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_scal"
  p_THDoubleBlas_scal :: FunPtr (CLong -> CDouble -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_copy : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_copy"
  p_THDoubleBlas_copy :: FunPtr (CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_axpy : Pointer to function n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_axpy"
  p_THDoubleBlas_axpy :: FunPtr (CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_dot : Pointer to function n x incx y incy -> real
foreign import ccall unsafe "THBlas.h &THDoubleBlas_dot"
  p_THDoubleBlas_dot :: FunPtr (CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble)

-- |p_THDoubleBlas_gemv : Pointer to function trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_gemv"
  p_THDoubleBlas_gemv :: FunPtr (CChar -> CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_ger : Pointer to function m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_ger"
  p_THDoubleBlas_ger :: FunPtr (CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> IO ())

-- |p_THDoubleBlas_gemm : Pointer to function transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h &THDoubleBlas_gemm"
  p_THDoubleBlas_gemm :: FunPtr (CChar -> CChar -> CLong -> CLong -> CLong -> CDouble -> Ptr CDouble -> CLong -> Ptr CDouble -> CLong -> CDouble -> Ptr CDouble -> CLong -> IO ())