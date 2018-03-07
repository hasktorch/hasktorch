{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.Lapack
  ( c_gesv
  , c_trtrs
  , c_gels
  , c_syev
  , c_geev
  , c_gesvd
  , c_getrf
  , c_getrs
  , c_getri
  , c_potrf
  , c_potri
  , c_potrs
  , c_pstrf
  , c_geqrf
  , c_orgqr
  , c_ormqr
  , p_gesv
  , p_trtrs
  , p_gels
  , p_syev
  , p_geev
  , p_gesvd
  , p_getrf
  , p_getrs
  , p_getri
  , p_potrf
  , p_potri
  , p_potrs
  , p_pstrf
  , p_geqrf
  , p_orgqr
  , p_ormqr
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesv"
  c_gesv :: CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_trtrs :  uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_trtrs"
  c_trtrs :: CChar -> CChar -> CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_gels :  trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gels"
  c_gels :: CChar -> CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_syev :  jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_syev"
  c_syev :: CChar -> CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_geev :  jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_geev"
  c_geev :: CChar -> CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_gesvd :  jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_gesvd"
  c_gesvd :: CChar -> CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_getrf :  m n a lda ipiv info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrf"
  c_getrf :: CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CInt) -> IO (())

-- | c_getrs :  trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_getrs"
  c_getrs :: CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_getri :  n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_getri"
  c_getri :: CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_potrf :  uplo n a lda info -> void
foreign import ccall "THLapack.h THDoubleLapack_potrf"
  c_potrf :: CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_potri :  uplo n a lda info -> void
foreign import ccall "THLapack.h THDoubleLapack_potri"
  c_potri :: CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_potrs :  uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h THDoubleLapack_potrs"
  c_potrs :: CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_pstrf :  uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h THDoubleLapack_pstrf"
  c_pstrf :: CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CInt) -> CDouble -> Ptr (CDouble) -> Ptr (CInt) -> IO (())

-- | c_geqrf :  m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_geqrf"
  c_geqrf :: CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_orgqr :  m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_orgqr"
  c_orgqr :: CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | c_ormqr :  side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h THDoubleLapack_ormqr"
  c_ormqr :: CChar -> CChar -> CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (())

-- | p_gesv : Pointer to function : n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gesv"
  p_gesv :: FunPtr (CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_trtrs : Pointer to function : uplo trans diag n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_trtrs"
  p_trtrs :: FunPtr (CChar -> CChar -> CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_gels : Pointer to function : trans m n nrhs a lda b ldb work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gels"
  p_gels :: FunPtr (CChar -> CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_syev : Pointer to function : jobz uplo n a lda w work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_syev"
  p_syev :: FunPtr (CChar -> CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_geev : Pointer to function : jobvl jobvr n a lda wr wi vl ldvl vr ldvr work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_geev"
  p_geev :: FunPtr (CChar -> CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_gesvd : Pointer to function : jobu jobvt m n a lda s u ldu vt ldvt work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_gesvd"
  p_gesvd :: FunPtr (CChar -> CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_getrf : Pointer to function : m n a lda ipiv info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getrf"
  p_getrf :: FunPtr (CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CInt) -> IO (()))

-- | p_getrs : Pointer to function : trans n nrhs a lda ipiv b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getrs"
  p_getrs :: FunPtr (CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_getri : Pointer to function : n a lda ipiv work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_getri"
  p_getri :: FunPtr (CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_potrf : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potrf"
  p_potrf :: FunPtr (CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_potri : Pointer to function : uplo n a lda info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potri"
  p_potri :: FunPtr (CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_potrs : Pointer to function : uplo n nrhs a lda b ldb info -> void
foreign import ccall "THLapack.h &THDoubleLapack_potrs"
  p_potrs :: FunPtr (CChar -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_pstrf : Pointer to function : uplo n a lda piv rank tol work info -> void
foreign import ccall "THLapack.h &THDoubleLapack_pstrf"
  p_pstrf :: FunPtr (CChar -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> Ptr (CInt) -> CDouble -> Ptr (CDouble) -> Ptr (CInt) -> IO (()))

-- | p_geqrf : Pointer to function : m n a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_geqrf"
  p_geqrf :: FunPtr (CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_orgqr : Pointer to function : m n k a lda tau work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_orgqr"
  p_orgqr :: FunPtr (CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))

-- | p_ormqr : Pointer to function : side trans m n k a lda tau c ldc work lwork info -> void
foreign import ccall "THLapack.h &THDoubleLapack_ormqr"
  p_ormqr :: FunPtr (CChar -> CChar -> CInt -> CInt -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> Ptr (CDouble) -> CInt -> Ptr (CDouble) -> CInt -> Ptr (CInt) -> IO (()))