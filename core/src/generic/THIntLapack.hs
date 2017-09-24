{-# LANGUAGE ForeignFunctionInterface #-}

module THIntLapack (
    c_THIntLapack_gesv,
    c_THIntLapack_trtrs,
    c_THIntLapack_gels,
    c_THIntLapack_syev,
    c_THIntLapack_geev,
    c_THIntLapack_gesvd,
    c_THIntLapack_getrf,
    c_THIntLapack_getrs,
    c_THIntLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THIntLapack_gesv"
  c_THIntLapack_gesv :: CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THIntLapack_trtrs"
  c_THIntLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THIntLapack_gels"
  c_THIntLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THIntLapack_syev"
  c_THIntLapack_syev :: CChar -> CChar -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THIntLapack_geev"
  c_THIntLapack_geev :: CChar -> CChar -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THIntLapack_gesvd"
  c_THIntLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_getrf : m n a lda ipiv info -> void
foreign import ccall "THLapack.h THIntLapack_getrf"
  c_THIntLapack_getrf :: CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THIntLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THIntLapack_getrs"
  c_THIntLapack_getrs :: CChar -> CInt -> CInt -> Ptr CInt -> CInt -> CIntPtr -> Ptr CInt -> CInt -> CIntPtr -> IO ()

-- |c_THIntLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THIntLapack_getri"
  c_THIntLapack_getri :: CInt -> Ptr CInt -> CInt -> CIntPtr -> Ptr CInt -> CInt -> CIntPtr -> IO ()