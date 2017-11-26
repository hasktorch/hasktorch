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
import Data.Word
import Data.Int

-- |c_THFloatBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_swap"
  c_THFloatBlas_swap :: CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THFloatBlas_scal"
  c_THFloatBlas_scal :: CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_copy"
  c_THFloatBlas_copy :: CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THFloatBlas_axpy"
  c_THFloatBlas_axpy :: CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THFloatBlas_dot"
  c_THFloatBlas_dot :: CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat

-- |c_THFloatBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THFloatBlas_gemv"
  c_THFloatBlas_gemv :: CChar -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THFloatBlas_ger"
  c_THFloatBlas_ger :: CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ()

-- |c_THFloatBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THFloatBlas_gemm"
  c_THFloatBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ()

-- |p_THFloatBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THFloatBlas_swap"
  p_THFloatBlas_swap :: FunPtr (CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THFloatBlas_scal"
  p_THFloatBlas_scal :: FunPtr (CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THFloatBlas_copy"
  p_THFloatBlas_copy :: FunPtr (CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THFloatBlas_axpy"
  p_THFloatBlas_axpy :: FunPtr (CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THFloatBlas_dot"
  p_THFloatBlas_dot :: FunPtr (CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat)

-- |p_THFloatBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THFloatBlas_gemv"
  p_THFloatBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THFloatBlas_ger"
  p_THFloatBlas_ger :: FunPtr (CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ())

-- |p_THFloatBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THFloatBlas_gemm"
  p_THFloatBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ())