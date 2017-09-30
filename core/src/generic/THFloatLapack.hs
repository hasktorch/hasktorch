{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatLapack (
    c_THFloatLapack_gesv,
    c_THFloatLapack_trtrs,
    c_THFloatLapack_gels,
    c_THFloatLapack_syev,
    c_THFloatLapack_geev,
    c_THFloatLapack_gesvd,
    c_THFloatLapack_getrf,
    c_THFloatLapack_getrs,
    c_THFloatLapack_getri,
    p_THFloatLapack_gesv,
    p_THFloatLapack_trtrs,
    p_THFloatLapack_gels,
    p_THFloatLapack_syev,
    p_THFloatLapack_geev,
    p_THFloatLapack_gesvd,
    p_THFloatLapack_getrf,
    p_THFloatLapack_getrs,
    p_THFloatLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_gesv"
  c_THFloatLapack_gesv :: CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_trtrs"
  c_THFloatLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_gels"
  c_THFloatLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_syev"
  c_THFloatLapack_syev :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_geev"
  c_THFloatLapack_geev :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_gesvd"
  c_THFloatLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_getrf : m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_getrf"
  c_THFloatLapack_getrf :: CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THFloatLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_getrs"
  c_THFloatLapack_getrs :: CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |c_THFloatLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h THFloatLapack_getri"
  c_THFloatLapack_getri :: CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- |p_THFloatLapack_gesv : Pointer to n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_gesv"
  p_THFloatLapack_gesv :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_trtrs : Pointer to uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_trtrs"
  p_THFloatLapack_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_gels : Pointer to trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_gels"
  p_THFloatLapack_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_syev : Pointer to jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_syev"
  p_THFloatLapack_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_geev : Pointer to jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_geev"
  p_THFloatLapack_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_gesvd : Pointer to jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_gesvd"
  p_THFloatLapack_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_getrf : Pointer to m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_getrf"
  p_THFloatLapack_getrf :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> CIntPtr -> IO ())

-- |p_THFloatLapack_getrs : Pointer to trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_getrs"
  p_THFloatLapack_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- |p_THFloatLapack_getri : Pointer to n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h &THFloatLapack_getri"
  p_THFloatLapack_getri :: FunPtr (CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())