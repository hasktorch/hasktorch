{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.Lapack where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_gesv"
  c_gesv_ :: CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gesv_ with unused argument (for CTHState) to unify backpack signatures.
c_gesv = const c_gesv_

-- | c_trtrs :  uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_trtrs"
  c_trtrs_ :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
c_trtrs = const c_trtrs_

-- | c_gels :  trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_gels"
  c_gels_ :: CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gels_ with unused argument (for CTHState) to unify backpack signatures.
c_gels = const c_gels_

-- | c_syev :  jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_syev"
  c_syev_ :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_syev_ with unused argument (for CTHState) to unify backpack signatures.
c_syev = const c_syev_

-- | c_geev :  jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_geev"
  c_geev_ :: CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_geev_ with unused argument (for CTHState) to unify backpack signatures.
c_geev = const c_geev_

-- | c_gesvd :  jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_gesvd"
  c_gesvd_ :: CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd = const c_gesvd_

-- | c_getrf :  m n a lda ipiv info -> void
foreign import ccall "THLapack.h THFloatLapack_getrf"
  c_getrf_ :: CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CInt -> IO ()

-- | alias of c_getrf_ with unused argument (for CTHState) to unify backpack signatures.
c_getrf = const c_getrf_

-- | c_getrs :  trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_getrs"
  c_getrs_ :: CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_getrs_ with unused argument (for CTHState) to unify backpack signatures.
c_getrs = const c_getrs_

-- | c_getri :  n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_getri"
  c_getri_ :: CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_getri_ with unused argument (for CTHState) to unify backpack signatures.
c_getri = const c_getri_

-- | c_potrf :  uplo n a lda info -> void
foreign import ccall "THLapack.h THFloatLapack_potrf"
  c_potrf_ :: CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potrf_ with unused argument (for CTHState) to unify backpack signatures.
c_potrf = const c_potrf_

-- | c_potri :  uplo n a lda info -> void
foreign import ccall "THLapack.h THFloatLapack_potri"
  c_potri_ :: CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potri_ with unused argument (for CTHState) to unify backpack signatures.
c_potri = const c_potri_

-- | c_potrs :  uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THFloatLapack_potrs"
  c_potrs_ :: CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_potrs_ with unused argument (for CTHState) to unify backpack signatures.
c_potrs = const c_potrs_

-- | c_pstrf :  uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h THFloatLapack_pstrf"
  c_pstrf_ :: CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CInt -> CFloat -> Ptr CFloat -> Ptr CInt -> IO ()

-- | alias of c_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
c_pstrf = const c_pstrf_

-- | c_geqrf :  m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_geqrf"
  c_geqrf_ :: CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
c_geqrf = const c_geqrf_

-- | c_orgqr :  m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_orgqr"
  c_orgqr_ :: CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
c_orgqr = const c_orgqr_

-- | c_ormqr :  side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h THFloatLapack_ormqr"
  c_ormqr_ :: CChar -> CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ()

-- | alias of c_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
c_ormqr = const c_ormqr_

-- | p_gesv : Pointer to function : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_gesv"
  p_gesv_ :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_gesv_ with unused argument (for CTHState) to unify backpack signatures.
p_gesv = const p_gesv_

-- | p_trtrs : Pointer to function : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_trtrs"
  p_trtrs_ :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
p_trtrs = const p_trtrs_

-- | p_gels : Pointer to function : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_gels"
  p_gels_ :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_gels_ with unused argument (for CTHState) to unify backpack signatures.
p_gels = const p_gels_

-- | p_syev : Pointer to function : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_syev"
  p_syev_ :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_syev_ with unused argument (for CTHState) to unify backpack signatures.
p_syev = const p_syev_

-- | p_geev : Pointer to function : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_geev"
  p_geev_ :: FunPtr (CChar -> CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_geev_ with unused argument (for CTHState) to unify backpack signatures.
p_geev = const p_geev_

-- | p_gesvd : Pointer to function : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_gesvd"
  p_gesvd_ :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
p_gesvd = const p_gesvd_

-- | p_getrf : Pointer to function : m n a lda ipiv info -> void
foreign import ccall "THLapack.h &THFloatLapack_getrf"
  p_getrf_ :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CInt -> IO ())

-- | alias of p_getrf_ with unused argument (for CTHState) to unify backpack signatures.
p_getrf = const p_getrf_

-- | p_getrs : Pointer to function : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_getrs"
  p_getrs_ :: FunPtr (CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_getrs_ with unused argument (for CTHState) to unify backpack signatures.
p_getrs = const p_getrs_

-- | p_getri : Pointer to function : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_getri"
  p_getri_ :: FunPtr (CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_getri_ with unused argument (for CTHState) to unify backpack signatures.
p_getri = const p_getri_

-- | p_potrf : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THFloatLapack_potrf"
  p_potrf_ :: FunPtr (CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_potrf_ with unused argument (for CTHState) to unify backpack signatures.
p_potrf = const p_potrf_

-- | p_potri : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THFloatLapack_potri"
  p_potri_ :: FunPtr (CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_potri_ with unused argument (for CTHState) to unify backpack signatures.
p_potri = const p_potri_

-- | p_potrs : Pointer to function : uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THFloatLapack_potrs"
  p_potrs_ :: FunPtr (CChar -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_potrs_ with unused argument (for CTHState) to unify backpack signatures.
p_potrs = const p_potrs_

-- | p_pstrf : Pointer to function : uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h &THFloatLapack_pstrf"
  p_pstrf_ :: FunPtr (CChar -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> Ptr CInt -> CFloat -> Ptr CFloat -> Ptr CInt -> IO ())

-- | alias of p_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
p_pstrf = const p_pstrf_

-- | p_geqrf : Pointer to function : m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_geqrf"
  p_geqrf_ :: FunPtr (CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
p_geqrf = const p_geqrf_

-- | p_orgqr : Pointer to function : m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_orgqr"
  p_orgqr_ :: FunPtr (CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
p_orgqr = const p_orgqr_

-- | p_ormqr : Pointer to function : side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h &THFloatLapack_ormqr"
  p_ormqr_ :: FunPtr (CChar -> CChar -> CInt -> CInt -> CInt -> Ptr CFloat -> CInt -> Ptr CFloat -> Ptr CFloat -> CInt -> Ptr CFloat -> CInt -> Ptr CInt -> IO ())

-- | alias of p_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
p_ormqr = const p_ormqr_