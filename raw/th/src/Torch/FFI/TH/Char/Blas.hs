{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Char.Blas where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_swap :  n x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_swap"
  c_swap_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_swap_ with unused argument (for CTHState) to unify backpack signatures.
c_swap :: Ptr C'THState -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()
c_swap = const c_swap_

-- | c_scal :  n a x incx -> void
foreign import ccall "THBlas.h THCharBlas_scal"
  c_scal_ :: CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_scal_ with unused argument (for CTHState) to unify backpack signatures.
c_scal :: Ptr C'THState -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()
c_scal = const c_scal_

-- | c_copy :  n x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_copy"
  c_copy_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_copy_ with unused argument (for CTHState) to unify backpack signatures.
c_copy :: Ptr C'THState -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()
c_copy = const c_copy_

-- | c_axpy :  n a x incx y incy -> void
foreign import ccall "THBlas.h THCharBlas_axpy"
  c_axpy_ :: CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_axpy_ with unused argument (for CTHState) to unify backpack signatures.
c_axpy :: Ptr C'THState -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()
c_axpy = const c_axpy_

-- | c_dot :  n x incx y incy -> real
foreign import ccall "THBlas.h THCharBlas_dot"
  c_dot_ :: CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO CChar

-- | alias of c_dot_ with unused argument (for CTHState) to unify backpack signatures.
c_dot :: Ptr C'THState -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO CChar
c_dot = const c_dot_

-- | c_gemv :  trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h THCharBlas_gemv"
  c_gemv_ :: CChar -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_gemv_ with unused argument (for CTHState) to unify backpack signatures.
c_gemv :: Ptr C'THState -> CChar -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()
c_gemv = const c_gemv_

-- | c_ger :  m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h THCharBlas_ger"
  c_ger_ :: CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_ger_ with unused argument (for CTHState) to unify backpack signatures.
c_ger :: Ptr C'THState -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ()
c_ger = const c_ger_

-- | c_gemm :  transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h THCharBlas_gemm"
  c_gemm_ :: CChar -> CChar -> CLLong -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()

-- | alias of c_gemm_ with unused argument (for CTHState) to unify backpack signatures.
c_gemm :: Ptr C'THState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ()
c_gemm = const c_gemm_

-- | p_swap : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_swap"
  p_swap :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | p_scal : Pointer to function : n a x incx -> void
foreign import ccall "THBlas.h &THCharBlas_scal"
  p_scal :: FunPtr (CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())

-- | p_copy : Pointer to function : n x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_copy"
  p_copy :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | p_axpy : Pointer to function : n a x incx y incy -> void
foreign import ccall "THBlas.h &THCharBlas_axpy"
  p_axpy :: FunPtr (CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | p_dot : Pointer to function : n x incx y incy -> real
foreign import ccall "THBlas.h &THCharBlas_dot"
  p_dot :: FunPtr (CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO CChar)

-- | p_gemv : Pointer to function : trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THBlas.h &THCharBlas_gemv"
  p_gemv :: FunPtr (CChar -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())

-- | p_ger : Pointer to function : m n alpha x incx y incy a lda -> void
foreign import ccall "THBlas.h &THCharBlas_ger"
  p_ger :: FunPtr (CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> IO ())

-- | p_gemm : Pointer to function : transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THBlas.h &THCharBlas_gemm"
  p_gemm :: FunPtr (CChar -> CChar -> CLLong -> CLLong -> CLLong -> CChar -> Ptr CChar -> CLLong -> Ptr CChar -> CLLong -> CChar -> Ptr CChar -> CLLong -> IO ())