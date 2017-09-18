{-# LANGUAGE ForeignFunctionInterface#-}

module THLongLapack (
    c_THLongLapack_gesv,
    c_THLongLapack_trtrs,
    c_THLongLapack_gels,
    c_THLongLapack_syev,
    c_THLongLapack_geev,
    c_THLongLapack_gesvd,
    c_THLongLapack_getrf,
    c_THLongLapack_getrs,
    c_THLongLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THLongLapack_gesv"
  c_THLongLapack_gesv :: CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THLongLapack_trtrs"
  c_THLongLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THLongLapack_gels"
  c_THLongLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THLongLapack_syev"
  c_THLongLapack_syev :: CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THLongLapack_geev"
  c_THLongLapack_geev :: CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THLongLapack_gesvd"
  c_THLongLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_getrf : m n a lda ipiv info -> void
foreign import ccall "THLapack.h THLongLapack_getrf"
  c_THLongLapack_getrf :: CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THLongLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THLongLapack_getrs"
  c_THLongLapack_getrs :: CChar -> CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THLongLapack_getri"
  c_THLongLapack_getri :: CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()