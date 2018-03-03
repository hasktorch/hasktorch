{-# LANGUAGE ForeignFunctionInterface #-}
module THHalfBlas
  ( c_swap
  , c_scal
  , c_copy
  , c_axpy
  , c_dot
  , c_gemv
  , c_ger
  , c_gemm
  , p_swap
  , p_scal
  , p_copy
  , p_axpy
  , p_dot
  , p_gemv
  , p_ger
  , p_gemm
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_swap"
  c_swap :: CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h THHalfBlas_scal"
  c_scal :: CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_copy"
  c_copy :: CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h THHalfBlas_axpy"
  c_axpy :: CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h THHalfBlas_dot"
  c_dot :: CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO (CTHHalf)

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THHalfBlas_gemv"
  c_gemv :: CChar -> CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THHalfBlas_ger"
  c_ger :: CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ()

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THHalfBlas_gemm"
  c_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ()

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_swap"
  p_swap :: FunPtr (CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THHalfBlas_scal"
  p_scal :: FunPtr (CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_copy"
  p_copy :: FunPtr (CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_axpy"
  p_axpy :: FunPtr (CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THHalfBlas_dot"
  p_dot :: FunPtr (CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO (CTHHalf))

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THHalfBlas_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THHalfBlas_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> IO ())

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THHalfBlas_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> Ptr CTHHalf -> CLLong -> CTHHalf -> Ptr CTHHalf -> CLLong -> IO ())