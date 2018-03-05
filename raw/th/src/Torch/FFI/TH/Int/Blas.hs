{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Int.Blas
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
foreign import ccall "THBlas.h c_THBlasInt_swap"
  c_swap :: CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (())

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h c_THBlasInt_scal"
  c_scal :: CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (())

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h c_THBlasInt_copy"
  c_copy :: CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (())

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h c_THBlasInt_axpy"
  c_axpy :: CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (())

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h c_THBlasInt_dot"
  c_dot :: CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (CInt)

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h c_THBlasInt_gemv"
  c_gemv :: CChar -> CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (())

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h c_THBlasInt_ger"
  c_ger :: CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (())

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h c_THBlasInt_gemm"
  c_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (())

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasInt_swap"
  p_swap :: FunPtr (CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &p_THBlasInt_scal"
  p_scal :: FunPtr (CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasInt_copy"
  p_copy :: FunPtr (CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasInt_axpy"
  p_axpy :: FunPtr (CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &p_THBlasInt_dot"
  p_dot :: FunPtr (CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (CInt))

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &p_THBlasInt_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &p_THBlasInt_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> IO (()))

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &p_THBlasInt_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> Ptr (CInt) -> CLLong -> CInt -> Ptr (CInt) -> CLLong -> IO (()))