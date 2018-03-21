{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Blas where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH
import Torch.Types.THC

-- | c_THCudaBlas_Sdot :  state n x incx y incy -> float
foreign import ccall "THCBlas.h THCudaBlas_Sdot"
  c_THCudaBlas_Sdot :: Ptr C'THCState -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO CFloat

-- | c_THCudaBlas_Ddot :  state n x incx y incy -> double
foreign import ccall "THCBlas.h THCudaBlas_Ddot"
  c_THCudaBlas_Ddot :: Ptr C'THCState -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO CDouble

-- | c_THCudaBlas_Sgemv :  state trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THCBlas.h THCudaBlas_Sgemv"
  c_THCudaBlas_Sgemv :: Ptr C'THCState -> CChar -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ()

-- | c_THCudaBlas_Dgemv :  state trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THCBlas.h THCudaBlas_Dgemv"
  c_THCudaBlas_Dgemv :: Ptr C'THCState -> CChar -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ()

-- | c_THCudaBlas_Sger :  state m n alpha x incx y incy a lda -> void
foreign import ccall "THCBlas.h THCudaBlas_Sger"
  c_THCudaBlas_Sger :: Ptr C'THCState -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ()

-- | c_THCudaBlas_Dger :  state m n alpha x incx y incy a lda -> void
foreign import ccall "THCBlas.h THCudaBlas_Dger"
  c_THCudaBlas_Dger :: Ptr C'THCState -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ()

-- | c_THCudaBlas_Sgemm :  state transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THCBlas.h THCudaBlas_Sgemm"
  c_THCudaBlas_Sgemm :: Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ()

-- | c_THCudaBlas_Dgemm :  state transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THCBlas.h THCudaBlas_Dgemm"
  c_THCudaBlas_Dgemm :: Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ()

-- | c_THCudaBlas_SgemmStridedBatched :  state transa transb m n k alpha a lda strideA b ldb strideB beta c ldc strideC batchCount -> void
foreign import ccall "THCBlas.h THCudaBlas_SgemmStridedBatched"
  c_THCudaBlas_SgemmStridedBatched :: Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> CLLong -> Ptr CFloat -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_THCudaBlas_DgemmStridedBatched :  state transa transb m n k alpha a lda strideA b ldb strideB beta c ldc strideC batchCount -> void
foreign import ccall "THCBlas.h THCudaBlas_DgemmStridedBatched"
  c_THCudaBlas_DgemmStridedBatched :: Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> CLLong -> Ptr CDouble -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> CLLong -> CLLong -> IO ()

-- | c_THCudaBlas_Sgetrf :  state n a lda pivot info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Sgetrf"
  c_THCudaBlas_Sgetrf :: Ptr C'THCState -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> IO ()

-- | c_THCudaBlas_Dgetrf :  state n a lda pivot info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Dgetrf"
  c_THCudaBlas_Dgetrf :: Ptr C'THCState -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> IO ()

-- | c_THCudaBlas_Sgetrs :  state transa n nrhs a lda pivot b ldb info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Sgetrs"
  c_THCudaBlas_Sgetrs :: Ptr C'THCState -> CChar -> CInt -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> CInt -> IO ()

-- | c_THCudaBlas_Dgetrs :  state transa n nrhs a lda pivot b ldb info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Dgetrs"
  c_THCudaBlas_Dgetrs :: Ptr C'THCState -> CChar -> CInt -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> CInt -> IO ()

-- | c_THCudaBlas_Sgetri :  state n a lda pivot c ldc info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Sgetri"
  c_THCudaBlas_Sgetri :: Ptr C'THCState -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> CInt -> IO ()

-- | c_THCudaBlas_Dgetri :  state n a lda pivot c ldc info batchSize -> void
foreign import ccall "THCBlas.h THCudaBlas_Dgetri"
  c_THCudaBlas_Dgetri :: Ptr C'THCState -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> CInt -> IO ()

-- | p_THCudaBlas_Sdot : Pointer to function : state n x incx y incy -> float
foreign import ccall "THCBlas.h &THCudaBlas_Sdot"
  p_THCudaBlas_Sdot :: FunPtr (Ptr C'THCState -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO CFloat)

-- | p_THCudaBlas_Ddot : Pointer to function : state n x incx y incy -> double
foreign import ccall "THCBlas.h &THCudaBlas_Ddot"
  p_THCudaBlas_Ddot :: FunPtr (Ptr C'THCState -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO CDouble)

-- | p_THCudaBlas_Sgemv : Pointer to function : state trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sgemv"
  p_THCudaBlas_Sgemv :: FunPtr (Ptr C'THCState -> CChar -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ())

-- | p_THCudaBlas_Dgemv : Pointer to function : state trans m n alpha a lda x incx beta y incy -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dgemv"
  p_THCudaBlas_Dgemv :: FunPtr (Ptr C'THCState -> CChar -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ())

-- | p_THCudaBlas_Sger : Pointer to function : state m n alpha x incx y incy a lda -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sger"
  p_THCudaBlas_Sger :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> IO ())

-- | p_THCudaBlas_Dger : Pointer to function : state m n alpha x incx y incy a lda -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dger"
  p_THCudaBlas_Dger :: FunPtr (Ptr C'THCState -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> IO ())

-- | p_THCudaBlas_Sgemm : Pointer to function : state transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sgemm"
  p_THCudaBlas_Sgemm :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> Ptr CFloat -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> IO ())

-- | p_THCudaBlas_Dgemm : Pointer to function : state transa transb m n k alpha a lda b ldb beta c ldc -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dgemm"
  p_THCudaBlas_Dgemm :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> Ptr CDouble -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> IO ())

-- | p_THCudaBlas_SgemmStridedBatched : Pointer to function : state transa transb m n k alpha a lda strideA b ldb strideB beta c ldc strideC batchCount -> void
foreign import ccall "THCBlas.h &THCudaBlas_SgemmStridedBatched"
  p_THCudaBlas_SgemmStridedBatched :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> CLLong -> Ptr CFloat -> CLLong -> CLLong -> CFloat -> Ptr CFloat -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_THCudaBlas_DgemmStridedBatched : Pointer to function : state transa transb m n k alpha a lda strideA b ldb strideB beta c ldc strideC batchCount -> void
foreign import ccall "THCBlas.h &THCudaBlas_DgemmStridedBatched"
  p_THCudaBlas_DgemmStridedBatched :: FunPtr (Ptr C'THCState -> CChar -> CChar -> CLLong -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> CLLong -> Ptr CDouble -> CLLong -> CLLong -> CDouble -> Ptr CDouble -> CLLong -> CLLong -> CLLong -> IO ())

-- | p_THCudaBlas_Sgetrf : Pointer to function : state n a lda pivot info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sgetrf"
  p_THCudaBlas_Sgetrf :: FunPtr (Ptr C'THCState -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> IO ())

-- | p_THCudaBlas_Dgetrf : Pointer to function : state n a lda pivot info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dgetrf"
  p_THCudaBlas_Dgetrf :: FunPtr (Ptr C'THCState -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> IO ())

-- | p_THCudaBlas_Sgetrs : Pointer to function : state transa n nrhs a lda pivot b ldb info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sgetrs"
  p_THCudaBlas_Sgetrs :: FunPtr (Ptr C'THCState -> CChar -> CInt -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> CInt -> IO ())

-- | p_THCudaBlas_Dgetrs : Pointer to function : state transa n nrhs a lda pivot b ldb info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dgetrs"
  p_THCudaBlas_Dgetrs :: FunPtr (Ptr C'THCState -> CChar -> CInt -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> CInt -> IO ())

-- | p_THCudaBlas_Sgetri : Pointer to function : state n a lda pivot c ldc info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Sgetri"
  p_THCudaBlas_Sgetri :: FunPtr (Ptr C'THCState -> CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> Ptr (Ptr CFloat) -> CInt -> Ptr CInt -> CInt -> IO ())

-- | p_THCudaBlas_Dgetri : Pointer to function : state n a lda pivot c ldc info batchSize -> void
foreign import ccall "THCBlas.h &THCudaBlas_Dgetri"
  p_THCudaBlas_Dgetri :: FunPtr (Ptr C'THCState -> CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> Ptr (Ptr CDouble) -> CInt -> Ptr CInt -> CInt -> IO ())