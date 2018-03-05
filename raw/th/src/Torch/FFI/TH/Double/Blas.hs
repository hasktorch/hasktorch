{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Blas
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
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h c_THBlasDouble_swap"
  c_swap :: CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h c_THBlasDouble_scal"
  c_scal :: CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h c_THBlasDouble_copy"
  c_copy :: CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h c_THBlasDouble_axpy"
  c_axpy :: CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h c_THBlasDouble_dot"
  c_dot :: CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (CDouble)

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h c_THBlasDouble_gemv"
  c_gemv :: CChar -> CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h c_THBlasDouble_ger"
  c_ger :: CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (())

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h c_THBlasDouble_gemm"
  c_gemm :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (())

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasDouble_swap"
  p_swap :: FunPtr (CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &p_THBlasDouble_scal"
  p_scal :: FunPtr (CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasDouble_copy"
  p_copy :: FunPtr (CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &p_THBlasDouble_axpy"
  p_axpy :: FunPtr (CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &p_THBlasDouble_dot"
  p_dot :: FunPtr (CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (CDouble))

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &p_THBlasDouble_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &p_THBlasDouble_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> IO (()))

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &p_THBlasDouble_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> Ptr (CDouble) -> CLLong -> CDouble -> Ptr (CDouble) -> CLLong -> IO (()))