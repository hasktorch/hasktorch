{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatBlas (
    c_THFloatBlas_swap,
    c_THFloatBlas_scal,
    c_THFloatBlas_copy,
    c_THFloatBlas_axpy,
    c_THFloatBlas_dot,
    c_THFloatBlas_gemv,
    c_THFloatBlas_ger,
    c_THFloatBlas_gemm,
    p_THFloatBlas_swap,
    p_THFloatBlas_scal,
    p_THFloatBlas_copy,
    p_THFloatBlas_axpy,
    p_THFloatBlas_dot,
    p_THFloatBlas_gemv,
    p_THFloatBlas_ger,
    p_THFloatBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatBlas_swap : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_swap"
  c_THFloatBlas_swap :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_scal : n a x incx -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_scal"
  c_THFloatBlas_scal :: CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_copy : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_copy"
  c_THFloatBlas_copy :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_axpy : n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_axpy"
  c_THFloatBlas_axpy :: CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_dot : n x incx y incy -> real
foreign import ccall unsafe "THBlas.h THFloatBlas_dot"
  c_THFloatBlas_dot :: CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat

-- |c_THFloatBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_gemv"
  c_THFloatBlas_gemv :: CChar -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_ger"
  c_THFloatBlas_ger :: CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ()

-- |c_THFloatBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h THFloatBlas_gemm"
  c_THFloatBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ()

-- |p_THFloatBlas_swap : Pointer to n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_swap"
  p_THFloatBlas_swap :: FunPtr (CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_scal : Pointer to n a x incx -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_scal"
  p_THFloatBlas_scal :: FunPtr (CLong -> CFloat -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_copy : Pointer to n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_copy"
  p_THFloatBlas_copy :: FunPtr (CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_axpy : Pointer to n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_axpy"
  p_THFloatBlas_axpy :: FunPtr (CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_dot : Pointer to n x incx y incy -> real
foreign import ccall unsafe "THBlas.h &THFloatBlas_dot"
  p_THFloatBlas_dot :: FunPtr (CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat)

-- |p_THFloatBlas_gemv : Pointer to trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_gemv"
  p_THFloatBlas_gemv :: FunPtr (CChar -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_ger : Pointer to m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_ger"
  p_THFloatBlas_ger :: FunPtr (CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> IO ())

-- |p_THFloatBlas_gemm : Pointer to transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h &THFloatBlas_gemm"
  p_THFloatBlas_gemm :: FunPtr (CChar -> CChar -> CLong -> CLong -> CLong -> CFloat -> Ptr CFloat -> CLong -> Ptr CFloat -> CLong -> CFloat -> Ptr CFloat -> CLong -> IO ())