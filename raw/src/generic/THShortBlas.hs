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
import Data.Word
import Data.Int

-- |c_THShortBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THShortBlas_swap"
  c_THShortBlas_swap :: CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THShortBlas_scal"
  c_THShortBlas_scal :: CLLong -> CShort -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THShortBlas_copy"
  c_THShortBlas_copy :: CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THShortBlas_axpy"
  c_THShortBlas_axpy :: CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THShortBlas_dot"
  c_THShortBlas_dot :: CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort

-- |c_THShortBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THShortBlas_gemv"
  c_THShortBlas_gemv :: CChar -> CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THShortBlas_ger"
  c_THShortBlas_ger :: CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ()

-- |c_THShortBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THShortBlas_gemm"
  c_THShortBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort -> Ptr CShort -> CLLong -> IO ()

-- |p_THShortBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THShortBlas_swap"
  p_THShortBlas_swap :: FunPtr (CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THShortBlas_scal"
  p_THShortBlas_scal :: FunPtr (CLLong -> CShort -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THShortBlas_copy"
  p_THShortBlas_copy :: FunPtr (CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THShortBlas_axpy"
  p_THShortBlas_axpy :: FunPtr (CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THShortBlas_dot"
  p_THShortBlas_dot :: FunPtr (CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort)

-- |p_THShortBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THShortBlas_gemv"
  p_THShortBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THShortBlas_ger"
  p_THShortBlas_ger :: FunPtr (CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> IO ())

-- |p_THShortBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THShortBlas_gemm"
  p_THShortBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CShort -> Ptr CShort -> CLLong -> Ptr CShort -> CLLong -> CShort -> Ptr CShort -> CLLong -> IO ())