{-# LANGUAGE ForeignFunctionInterface #-}

module THDoubleLapack (
    c_THDoubleLapack_gesv,
    c_THDoubleLapack_trtrs,
    c_THDoubleLapack_gels,
    c_THDoubleLapack_syev,
    c_THDoubleLapack_geev,
    c_THDoubleLapack_gesvd,
    c_THDoubleLapack_getrf,
    c_THDoubleLapack_getrs,
    c_THDoubleLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesv"
  c_THDoubleLapack_gesv :: CInt -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_trtrs"
  c_THDoubleLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gels"
  c_THDoubleLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_syev"
  c_THDoubleLapack_syev :: CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_geev"
  c_THDoubleLapack_geev :: CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesvd"
  c_THDoubleLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_getrf : m n a lda ipiv info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrf"
  c_THDoubleLapack_getrf :: CInt -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THDoubleLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrs"
  c_THDoubleLapack_getrs :: CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> CIntPtr -> Ptr CDouble -> CInt -> CIntPtr -> IO ()

-- |c_THDoubleLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_getri"
  c_THDoubleLapack_getri :: CInt -> Ptr CDouble -> CInt -> CIntPtr -> Ptr CDouble -> CInt -> CIntPtr -> IO ()