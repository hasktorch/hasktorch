{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Lapack where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_gesv :  n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesv"
  c_gesv_ :: CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gesv_ with unused argument (for CTHState) to unify backpack signatures.
c_gesv :: Ptr C'THState -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_gesv = const c_gesv_

-- | c_trtrs :  uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_trtrs"
  c_trtrs_ :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
c_trtrs :: Ptr C'THState -> CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_trtrs = const c_trtrs_

-- | c_gels :  trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gels"
  c_gels_ :: CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gels_ with unused argument (for CTHState) to unify backpack signatures.
c_gels :: Ptr C'THState -> CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_gels = const c_gels_

-- | c_syev :  jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_syev"
  c_syev_ :: CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_syev_ with unused argument (for CTHState) to unify backpack signatures.
c_syev :: Ptr C'THState -> CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_syev = const c_syev_

-- | c_geev :  jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_geev"
  c_geev_ :: CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_geev_ with unused argument (for CTHState) to unify backpack signatures.
c_geev :: Ptr C'THState -> CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_geev = const c_geev_

-- | c_gesvd :  jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesvd"
  c_gesvd_ :: CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd :: Ptr C'THState -> CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_gesvd = const c_gesvd_

-- | c_getrf :  m n a lda ipiv info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrf"
  c_getrf_ :: CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> IO ()

-- | alias of c_getrf_ with unused argument (for CTHState) to unify backpack signatures.
c_getrf :: Ptr C'THState -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> IO ()
c_getrf = const c_getrf_

-- | c_getrs :  trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrs"
  c_getrs_ :: CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_getrs_ with unused argument (for CTHState) to unify backpack signatures.
c_getrs :: Ptr C'THState -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_getrs = const c_getrs_

-- | c_getri :  n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_getri"
  c_getri_ :: CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_getri_ with unused argument (for CTHState) to unify backpack signatures.
c_getri :: Ptr C'THState -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_getri = const c_getri_

-- | c_potrf :  uplo n a lda info -> void
foreign import ccall "THLapack.h THDoubleLapack_potrf"
  c_potrf_ :: CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potrf_ with unused argument (for CTHState) to unify backpack signatures.
c_potrf :: Ptr C'THState -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_potrf = const c_potrf_

-- | c_potri :  uplo n a lda info -> void
foreign import ccall "THLapack.h THDoubleLapack_potri"
  c_potri_ :: CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potri_ with unused argument (for CTHState) to unify backpack signatures.
c_potri :: Ptr C'THState -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_potri = const c_potri_

-- | c_potrs :  uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_potrs"
  c_potrs_ :: CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potrs_ with unused argument (for CTHState) to unify backpack signatures.
c_potrs :: Ptr C'THState -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_potrs = const c_potrs_

-- | c_pstrf :  uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h THDoubleLapack_pstrf"
  c_pstrf_ :: CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> CDouble -> Ptr CDouble -> Ptr CInt -> IO ()

-- | alias of c_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
c_pstrf :: Ptr C'THState -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> CDouble -> Ptr CDouble -> Ptr CInt -> IO ()
c_pstrf = const c_pstrf_

-- | c_geqrf :  m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_geqrf"
  c_geqrf_ :: CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
c_geqrf :: Ptr C'THState -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_geqrf = const c_geqrf_

-- | c_orgqr :  m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_orgqr"
  c_orgqr_ :: CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
c_orgqr :: Ptr C'THState -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_orgqr = const c_orgqr_

-- | c_ormqr :  side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_ormqr"
  c_ormqr_ :: CChar -> CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()

-- | alias of c_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
c_ormqr :: Ptr C'THState -> CChar -> CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ()
c_ormqr = const c_ormqr_

-- | p_gesv : Pointer to function : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gesv"
  p_gesv :: FunPtr (CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_trtrs : Pointer to function : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_trtrs"
  p_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_gels : Pointer to function : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gels"
  p_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_syev : Pointer to function : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_syev"
  p_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_geev : Pointer to function : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_geev"
  p_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_gesvd : Pointer to function : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gesvd"
  p_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_getrf : Pointer to function : m n a lda ipiv info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getrf"
  p_getrf :: FunPtr (CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> IO ())

-- | p_getrs : Pointer to function : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getrs"
  p_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_getri : Pointer to function : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getri"
  p_getri :: FunPtr (CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_potrf : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potrf"
  p_potrf :: FunPtr (CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_potri : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potri"
  p_potri :: FunPtr (CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_potrs : Pointer to function : uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potrs"
  p_potrs :: FunPtr (CChar -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_pstrf : Pointer to function : uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h &THDoubleLapack_pstrf"
  p_pstrf :: FunPtr (CChar -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> Ptr CInt -> CDouble -> Ptr CDouble -> Ptr CInt -> IO ())

-- | p_geqrf : Pointer to function : m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_geqrf"
  p_geqrf :: FunPtr (CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_orgqr : Pointer to function : m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_orgqr"
  p_orgqr :: FunPtr (CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())

-- | p_ormqr : Pointer to function : side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_ormqr"
  p_ormqr :: FunPtr (CChar -> CChar -> CInt -> CInt -> CInt -> Ptr CDouble -> CInt -> Ptr CDouble -> Ptr CDouble -> CInt -> Ptr CDouble -> CInt -> Ptr CInt -> IO ())