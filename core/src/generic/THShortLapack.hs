{-# LANGUAGE ForeignFunctionInterface #-}

module THShortLapack (
    c_THShortLapack_gesv,
    c_THShortLapack_trtrs,
    c_THShortLapack_gels,
    c_THShortLapack_syev,
    c_THShortLapack_geev,
    c_THShortLapack_gesvd,
    c_THShortLapack_getrf,
    c_THShortLapack_getrs,
    c_THShortLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_gesv"
  c_THShortLapack_gesv :: CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_trtrs"
  c_THShortLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_gels"
  c_THShortLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_syev"
  c_THShortLapack_syev :: CChar -> CChar -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_geev"
  c_THShortLapack_geev :: CChar -> CChar -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_gesvd"
  c_THShortLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_getrf : m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_getrf"
  c_THShortLapack_getrf :: CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THShortLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_getrs"
  c_THShortLapack_getrs :: CChar -> CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ()

-- |c_THShortLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h THShortLapack_getri"
  c_THShortLapack_getri :: CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ()