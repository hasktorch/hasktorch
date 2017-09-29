{-# LANGUAGE ForeignFunctionInterface #-}

module THByteLapack (
    c_THByteLapack_gesv,
    c_THByteLapack_trtrs,
    c_THByteLapack_gels,
    c_THByteLapack_syev,
    c_THByteLapack_geev,
    c_THByteLapack_gesvd,
    c_THByteLapack_getrf,
    c_THByteLapack_getrs,
    c_THByteLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THByteLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_gesv"
  c_THByteLapack_gesv :: CInt -> CInt -> Ptr CChar -> CInt -> CIntPtr -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_trtrs"
  c_THByteLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_gels"
  c_THByteLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_syev"
  c_THByteLapack_syev :: CChar -> CChar -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_geev"
  c_THByteLapack_geev :: CChar -> CChar -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> Ptr CChar -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_gesvd"
  c_THByteLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_getrf : m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_getrf"
  c_THByteLapack_getrf :: CInt -> CInt -> Ptr CChar -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THByteLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_getrs"
  c_THByteLapack_getrs :: CChar -> CInt -> CInt -> Ptr CChar -> CInt -> CIntPtr -> Ptr CChar -> CInt -> CIntPtr -> IO ()

-- |c_THByteLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h THByteLapack_getri"
  c_THByteLapack_getri :: CInt -> Ptr CChar -> CInt -> CIntPtr -> Ptr CChar -> CInt -> CIntPtr -> IO ()