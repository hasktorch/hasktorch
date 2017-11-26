{-# LANGUAGE ForeignFunctionInterface #-}

module THIntBlas (
    c_THIntBlas_swap,
    c_THIntBlas_scal,
    c_THIntBlas_copy,
    c_THIntBlas_axpy,
    c_THIntBlas_dot,
    c_THIntBlas_gemv,
    c_THIntBlas_ger,
    c_THIntBlas_gemm,
    p_THIntBlas_swap,
    p_THIntBlas_scal,
    p_THIntBlas_copy,
    p_THIntBlas_axpy,
    p_THIntBlas_dot,
    p_THIntBlas_gemv,
    p_THIntBlas_ger,
    p_THIntBlas_gemm) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- |c_THIntBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_swap"
  c_THIntBlas_swap :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THIntBlas_scal"
  c_THIntBlas_scal :: CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_copy"
  c_THIntBlas_copy :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_axpy"
  c_THIntBlas_axpy :: CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THIntBlas_dot"
  c_THIntBlas_dot :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt

-- |c_THIntBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THIntBlas_gemv"
  c_THIntBlas_gemv :: CChar -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THIntBlas_ger"
  c_THIntBlas_ger :: CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- |c_THIntBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THIntBlas_gemm"
  c_THIntBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- |p_THIntBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_swap"
  p_THIntBlas_swap :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THIntBlas_scal"
  p_THIntBlas_scal :: FunPtr (CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_copy"
  p_THIntBlas_copy :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_axpy"
  p_THIntBlas_axpy :: FunPtr (CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THIntBlas_dot"
  p_THIntBlas_dot :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt)

-- |p_THIntBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THIntBlas_gemv"
  p_THIntBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THIntBlas_ger"
  p_THIntBlas_ger :: FunPtr (CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- |p_THIntBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THIntBlas_gemm"
  p_THIntBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())