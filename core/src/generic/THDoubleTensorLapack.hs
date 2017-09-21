{-# LANGUAGE ForeignFunctionInterface#-}

module THDoubleTensorLapack (
    c_THDoubleTensor_gesv,
    c_THDoubleTensor_trtrs,
    c_THDoubleTensor_gels,
    c_THDoubleTensor_syev,
    c_THDoubleTensor_geev,
    c_THDoubleTensor_gesvd,
    c_THDoubleTensor_gesvd2,
    c_THDoubleTensor_getri,
    c_THDoubleTensor_potrf,
    c_THDoubleTensor_potrs,
    c_THDoubleTensor_potri,
    c_THDoubleTensor_qr,
    c_THDoubleTensor_geqrf,
    c_THDoubleTensor_orgqr,
    c_THDoubleTensor_ormqr,
    c_THDoubleTensor_pstrf,
    c_THDoubleTensor_btrifact,
    c_THDoubleTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesv"
  c_THDoubleTensor_gesv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_trtrs"
  c_THDoubleTensor_trtrs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> CChar -> CChar -> IO ()

-- |c_THDoubleTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gels"
  c_THDoubleTensor_gels :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_syev"
  c_THDoubleTensor_syev :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> CChar -> IO ()

-- |c_THDoubleTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_geev"
  c_THDoubleTensor_geev :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesvd"
  c_THDoubleTensor_gesvd :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesvd2"
  c_THDoubleTensor_gesvd2 :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_getri : ra_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_getri"
  c_THDoubleTensor_getri :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potrf"
  c_THDoubleTensor_potrf :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potrs"
  c_THDoubleTensor_potrs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_potri : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potri"
  c_THDoubleTensor_potri :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> IO ()

-- |c_THDoubleTensor_qr : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_qr"
  c_THDoubleTensor_qr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_geqrf"
  c_THDoubleTensor_geqrf :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_orgqr : ra_ a tau -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_orgqr"
  c_THDoubleTensor_orgqr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_ormqr"
  c_THDoubleTensor_ormqr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> CChar -> CChar -> IO ()

-- |c_THDoubleTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_pstrf"
  c_THDoubleTensor_pstrf :: (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> (Ptr CTHDoubleTensor) -> CChar -> CDouble -> IO ()

-- |c_THDoubleTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_btrifact"
  c_THDoubleTensor_btrifact :: (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_btrisolve"
  c_THDoubleTensor_btrisolve :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> IO ()