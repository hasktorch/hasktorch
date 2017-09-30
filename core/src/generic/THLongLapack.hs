{-# LANGUAGE ForeignFunctionInterface #-}

module THLongLapack (
    c_THLongLapack_gesv,
    c_THLongLapack_trtrs,
    c_THLongLapack_gels,
    c_THLongLapack_syev,
    c_THLongLapack_geev,
    c_THLongLapack_gesvd,
    c_THLongLapack_getrf,
    c_THLongLapack_getrs,
    c_THLongLapack_getri,
    p_THLongLapack_gesv,
    p_THLongLapack_trtrs,
    p_THLongLapack_gels,
    p_THLongLapack_syev,
    p_THLongLapack_geev,
    p_THLongLapack_gesvd,
    p_THLongLapack_getrf,
    p_THLongLapack_getrs,
    p_THLongLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_gesv"
  c_THLongLapack_gesv :: CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_trtrs"
  c_THLongLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_gels"
  c_THLongLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_syev"
  c_THLongLapack_syev :: CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_geev"
  c_THLongLapack_geev :: CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_gesvd"
  c_THLongLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_getrf : m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_getrf"
  c_THLongLapack_getrf :: CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THLongLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_getrs"
  c_THLongLapack_getrs :: CChar -> CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |c_THLongLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h THLongLapack_getri"
  c_THLongLapack_getri :: CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ()

-- |p_THLongLapack_gesv : Pointer to n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_gesv"
  p_THLongLapack_gesv :: FunPtr (CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_trtrs : Pointer to uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_trtrs"
  p_THLongLapack_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_gels : Pointer to trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_gels"
  p_THLongLapack_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_syev : Pointer to jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_syev"
  p_THLongLapack_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_geev : Pointer to jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_geev"
  p_THLongLapack_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_gesvd : Pointer to jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_gesvd"
  p_THLongLapack_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_getrf : Pointer to m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_getrf"
  p_THLongLapack_getrf :: FunPtr (CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> CIntPtr -> IO ())

-- |p_THLongLapack_getrs : Pointer to trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_getrs"
  p_THLongLapack_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ())

-- |p_THLongLapack_getri : Pointer to n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h &THLongLapack_getri"
  p_THLongLapack_getri :: FunPtr (CInt -> Ptr CLong -> CInt -> CIntPtr -> Ptr CLong -> CInt -> CIntPtr -> IO ())