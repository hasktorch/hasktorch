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
import Data.Word
import Data.Int

-- |c_THDoubleBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THDoubleBlas_swap"
  c_THDoubleBlas_swap :: CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THDoubleBlas_scal"
  c_THDoubleBlas_scal :: CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THDoubleBlas_copy"
  c_THDoubleBlas_copy :: CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THDoubleBlas_axpy"
  c_THDoubleBlas_axpy :: CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THDoubleBlas_dot"
  c_THDoubleBlas_dot :: CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble

-- |c_THDoubleBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THDoubleBlas_gemv"
  c_THDoubleBlas_gemv :: CChar -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THDoubleBlas_ger"
  c_THDoubleBlas_ger :: CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ()

-- |c_THDoubleBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THDoubleBlas_gemm"
  c_THDoubleBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ()

-- |p_THDoubleBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THDoubleBlas_swap"
  p_THDoubleBlas_swap :: FunPtr (CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THDoubleBlas_scal"
  p_THDoubleBlas_scal :: FunPtr (CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THDoubleBlas_copy"
  p_THDoubleBlas_copy :: FunPtr (CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THDoubleBlas_axpy"
  p_THDoubleBlas_axpy :: FunPtr (CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THDoubleBlas_dot"
  p_THDoubleBlas_dot :: FunPtr (CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble)

-- |p_THDoubleBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THDoubleBlas_gemv"
  p_THDoubleBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THDoubleBlas_ger"
  p_THDoubleBlas_ger :: FunPtr (CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ())

-- |p_THDoubleBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THDoubleBlas_gemm"
  p_THDoubleBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ())