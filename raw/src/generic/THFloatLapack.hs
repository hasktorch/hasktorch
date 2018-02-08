{-# LANGUAGE ForeignFunctionInterface #-}
module THFloatLapack
  ( c_gesv
  , c_trtrs
  , c_gels
  , c_syev
  , c_geev
  , c_gesvd
  , c_getrf
  , c_getrs
  , c_getri
  , p_gesv
  , p_trtrs
  , p_gels
  , p_syev
  , p_geev
  , p_gesvd
  , p_getrf
  , p_getrs
  , p_getri
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_gesv :  n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_gesv"
  c_gesv :: CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_trtrs :  uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_trtrs"
  c_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_gels :  trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_gels"
  c_gels :: CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_syev :  jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_syev"
  c_syev :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_geev :  jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_geev"
  c_geev :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_gesvd :  jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_gesvd"
  c_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_getrf :  m n a lda ipiv info -> void
foreign import ccall "THLapack.h THFloatLapack_getrf"
  c_getrf :: CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> CIntPtr -> IO ()

-- | c_getrs :  trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_getrs"
  c_getrs :: CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | c_getri :  n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_getri"
  c_getri :: CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ()

-- | p_gesv : Pointer to function : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_gesv"
  p_gesv :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_trtrs : Pointer to function : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_trtrs"
  p_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_gels : Pointer to function : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_gels"
  p_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_syev : Pointer to function : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_syev"
  p_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_geev : Pointer to function : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_geev"
  p_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_gesvd : Pointer to function : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_gesvd"
  p_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_getrf : Pointer to function : m n a lda ipiv info -> void
foreign import ccall "THLapack.h &THFloatLapack_getrf"
  p_getrf :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> CIntPtr -> IO ())

-- | p_getrs : Pointer to function : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_getrs"
  p_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())

-- | p_getri : Pointer to function : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_getri"
  p_getri :: FunPtr (CInt -> Ptr CFloat -> CInt -> CIntPtr -> Ptr CFloat -> CInt -> CIntPtr -> IO ())