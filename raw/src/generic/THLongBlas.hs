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
import Data.Word
import Data.Int

-- |c_THLongBlas_swap : n x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_swap"
  c_THLongBlas_swap :: CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_scal : n a x incx -> void
foreign import ccall "THBlas.h THLongBlas_scal"
  c_THLongBlas_scal :: CLLong -> CLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_copy : n x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_copy"
  c_THLongBlas_copy :: CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_axpy : n a x incx y incy -> void
foreign import ccall "THBlas.h THLongBlas_axpy"
  c_THLongBlas_axpy :: CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_dot : n x incx y incy -> real
foreign import ccall "THBlas.h THLongBlas_dot"
  c_THLongBlas_dot :: CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong

-- |c_THLongBlas_gemv : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THLongBlas_gemv"
  c_THLongBlas_gemv :: CChar -> CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_ger : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THLongBlas_ger"
  c_THLongBlas_ger :: CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ()

-- |c_THLongBlas_gemm : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THLongBlas_gemm"
  c_THLongBlas_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> IO ()

-- |p_THLongBlas_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THLongBlas_swap"
  p_THLongBlas_swap :: FunPtr (CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THLongBlas_scal"
  p_THLongBlas_scal :: FunPtr (CLLong -> CLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THLongBlas_copy"
  p_THLongBlas_copy :: FunPtr (CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THLongBlas_axpy"
  p_THLongBlas_axpy :: FunPtr (CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THLongBlas_dot"
  p_THLongBlas_dot :: FunPtr (CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong)

-- |p_THLongBlas_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THLongBlas_gemv"
  p_THLongBlas_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THLongBlas_ger"
  p_THLongBlas_ger :: FunPtr (CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> IO ())

-- |p_THLongBlas_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THLongBlas_gemm"
  p_THLongBlas_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> Ptr CLong -> CLLong -> CLong -> Ptr CLong -> CLLong -> IO ())