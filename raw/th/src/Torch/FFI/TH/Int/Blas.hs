{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Blas where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_swap"
  c_swap :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h THIntBlas_scal"
  c_scal :: CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_copy"
  c_copy :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h THIntBlas_axpy"
  c_axpy :: CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h THIntBlas_dot"
  c_dot :: CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO CInt

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THIntBlas_gemv"
  c_gemv :: CChar -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THIntBlas_ger"
  c_ger :: CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ()

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THIntBlas_gemm"
  c_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ()

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_swap"
  p_swap :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THIntBlas_scal"
  p_scal :: FunPtr (CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_copy"
  p_copy :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THIntBlas_axpy"
  p_axpy :: FunPtr (CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THIntBlas_dot"
  p_dot :: FunPtr (CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO CInt)

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THIntBlas_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THIntBlas_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> IO ())

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THIntBlas_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr CInt -> CLLong -> Ptr CInt -> CLLong -> CInt -> Ptr CInt -> CLLong -> IO ())