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
    c_THShortLapack_getri,
    p_THShortLapack_gesv,
    p_THShortLapack_trtrs,
    p_THShortLapack_gels,
    p_THShortLapack_syev,
    p_THShortLapack_geev,
    p_THShortLapack_gesvd,
    p_THShortLapack_getrf,
    p_THShortLapack_getrs,
    p_THShortLapack_getri) where

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

-- |p_THShortLapack_gesv : Pointer to n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_gesv"
  p_THShortLapack_gesv :: FunPtr (CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_trtrs : Pointer to uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_trtrs"
  p_THShortLapack_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_gels : Pointer to trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_gels"
  p_THShortLapack_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_syev : Pointer to jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_syev"
  p_THShortLapack_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_geev : Pointer to jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_geev"
  p_THShortLapack_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_gesvd : Pointer to jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_gesvd"
  p_THShortLapack_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_getrf : Pointer to m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_getrf"
  p_THShortLapack_getrf :: FunPtr (CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> CIntPtr -> IO ())

-- |p_THShortLapack_getrs : Pointer to trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_getrs"
  p_THShortLapack_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ())

-- |p_THShortLapack_getri : Pointer to n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h &THShortLapack_getri"
  p_THShortLapack_getri :: FunPtr (CInt -> Ptr CShort -> CInt -> CIntPtr -> Ptr CShort -> CInt -> CIntPtr -> IO ())