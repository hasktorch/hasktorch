{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfLapack (
    c_THHalfLapack_gesv,
    c_THHalfLapack_trtrs,
    c_THHalfLapack_gels,
    c_THHalfLapack_syev,
    c_THHalfLapack_geev,
    c_THHalfLapack_gesvd,
    c_THHalfLapack_getrf,
    c_THHalfLapack_getrs,
    c_THHalfLapack_getri) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfLapack_gesv : n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_gesv"
  c_THHalfLapack_gesv :: CInt -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_trtrs : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_trtrs"
  c_THHalfLapack_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_gels : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_gels"
  c_THHalfLapack_gels :: CChar -> CInt -> CInt -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_syev : jobz uplo n a lda w work lwork info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_syev"
  c_THHalfLapack_syev :: CChar -> CChar -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_geev : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_geev"
  c_THHalfLapack_geev :: CChar -> CChar -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> Ptr THHalf -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_gesvd : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_gesvd"
  c_THHalfLapack_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_getrf : m n a lda ipiv info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_getrf"
  c_THHalfLapack_getrf :: CInt -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- |c_THHalfLapack_getrs : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_getrs"
  c_THHalfLapack_getrs :: CChar -> CInt -> CInt -> Ptr THHalf -> CInt -> CIntPtr -> Ptr THHalf -> CInt -> CIntPtr -> IO ()

-- |c_THHalfLapack_getri : n a lda ipiv work lwork info -> void
foreign import ccall unsafe "THLapack.h THHalfLapack_getri"
  c_THHalfLapack_getri :: CInt -> Ptr THHalf -> CInt -> CIntPtr -> Ptr THHalf -> CInt -> CIntPtr -> IO ()