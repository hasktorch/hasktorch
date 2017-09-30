{-# LANGUAGE ForeignFunctionInterface #-}

module THLongBlas (
    c_THLongBlas_swap,
    c_THLongBlas_scal,
    c_THLongBlas_copy,
    c_THLongBlas_axpy,
    c_THLongBlas_dot,
    c_THLongBlas_gemv,
    c_THLongBlas_ger,
    c_THLongBlas_gemm,
    p_THLongBlas_swap,
    p_THLongBlas_scal,
    p_THLongBlas_copy,
    p_THLongBlas_axpy,
    p_THLongBlas_dot,
    p_THLongBlas_gemv,
    p_THLongBlas_ger,
    p_THLongBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongBlas_swap : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THLongBlas_swap"
  c_THLongBlas_swap :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_scal : n a x incx -> void
foreign import ccall unsafe "THBlas.h THLongBlas_scal"
  c_THLongBlas_scal :: CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_copy : n x incx y incy -> void
foreign import ccall unsafe "THBlas.h THLongBlas_copy"
  c_THLongBlas_copy :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_axpy : n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h THLongBlas_axpy"
  c_THLongBlas_axpy :: CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_dot : n x incx y incy -> real
foreign import ccall unsafe "THBlas.h THLongBlas_dot"
  c_THLongBlas_dot :: CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong

-- |c_THLongBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h THLongBlas_gemv"
  c_THLongBlas_gemv :: CChar -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h THLongBlas_ger"
  c_THLongBlas_ger :: CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |c_THLongBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h THLongBlas_gemm"
  c_THLongBlas_gemm :: CChar -> CChar -> CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ()

-- |p_THLongBlas_swap : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_swap"
  p_THLongBlas_swap :: FunPtr (CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_scal : Pointer to function n a x incx -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_scal"
  p_THLongBlas_scal :: FunPtr (CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_copy : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_copy"
  p_THLongBlas_copy :: FunPtr (CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_axpy : Pointer to function n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_axpy"
  p_THLongBlas_axpy :: FunPtr (CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_dot : Pointer to function n x incx y incy -> real
foreign import ccall unsafe "THBlas.h &THLongBlas_dot"
  p_THLongBlas_dot :: FunPtr (CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong)

-- |p_THLongBlas_gemv : Pointer to function trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_gemv"
  p_THLongBlas_gemv :: FunPtr (CChar -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_ger : Pointer to function m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_ger"
  p_THLongBlas_ger :: FunPtr (CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> IO ())

-- |p_THLongBlas_gemm : Pointer to function transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h &THLongBlas_gemm"
  p_THLongBlas_gemm :: FunPtr (CChar -> CChar -> CLong -> CLong -> CLong -> CLong -> Ptr CLong -> CLong -> Ptr CLong -> CLong -> CLong -> Ptr CLong -> CLong -> IO ())