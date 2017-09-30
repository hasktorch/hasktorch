{-# LANGUAGE ForeignFunctionInterface #-}

module THShortBlas (
    c_THShortBlas_swap,
    c_THShortBlas_scal,
    c_THShortBlas_copy,
    c_THShortBlas_axpy,
    c_THShortBlas_dot,
    c_THShortBlas_gemv,
    c_THShortBlas_ger,
    c_THShortBlas_gemm,
    p_THShortBlas_swap,
    p_THShortBlas_scal,
    p_THShortBlas_copy,
    p_THShortBlas_axpy,
    p_THShortBlas_dot,
    p_THShortBlas_gemv,
    p_THShortBlas_ger,
    p_THShortBlas_gemm) where

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

-- |p_THShortBlas_swap : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_swap"
  p_THShortBlas_swap :: FunPtr (CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_scal : Pointer to function n a x incx -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_scal"
  p_THShortBlas_scal :: FunPtr (CLong -> CShort -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_copy : Pointer to function n x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_copy"
  p_THShortBlas_copy :: FunPtr (CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_axpy : Pointer to function n a x incx y incy -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_axpy"
  p_THShortBlas_axpy :: FunPtr (CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_dot : Pointer to function n x incx y incy -> real
foreign import ccall unsafe "THBlas.h &THShortBlas_dot"
  p_THShortBlas_dot :: FunPtr (CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort)

-- |p_THShortBlas_gemv : Pointer to function trans m n alpha a lda x incx beta y incy -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_gemv"
  p_THShortBlas_gemv :: FunPtr (CChar -> CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_ger : Pointer to function m n alpha x incx y incy a lda -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_ger"
  p_THShortBlas_ger :: FunPtr (CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> IO ())

-- |p_THShortBlas_gemm : Pointer to function transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall unsafe "THBlas.h &THShortBlas_gemm"
  p_THShortBlas_gemm :: FunPtr (CChar -> CChar -> CLong -> CLong -> CLong -> CShort -> Ptr CShort -> CLong -> Ptr CShort -> CLong -> CShort -> Ptr CShort -> CLong -> IO ())