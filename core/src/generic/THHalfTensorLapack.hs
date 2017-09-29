{-# LANGUAGE ForeignFunctionInterface #-}

module THHalfTensorLapack (
    c_THHalfTensor_gesv,
    c_THHalfTensor_trtrs,
    c_THHalfTensor_gels,
    c_THHalfTensor_syev,
    c_THHalfTensor_geev,
    c_THHalfTensor_gesvd,
    c_THHalfTensor_gesvd2,
    c_THHalfTensor_getri,
    c_THHalfTensor_potrf,
    c_THHalfTensor_potrs,
    c_THHalfTensor_potri,
    c_THHalfTensor_qr,
    c_THHalfTensor_geqrf,
    c_THHalfTensor_orgqr,
    c_THHalfTensor_ormqr,
    c_THHalfTensor_pstrf,
    c_THHalfTensor_btrifact,
    c_THHalfTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THHalfTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_gesv"
  c_THHalfTensor_gesv :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_trtrs"
  c_THHalfTensor_trtrs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_gels"
  c_THHalfTensor_gels :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_syev"
  c_THHalfTensor_syev :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_geev"
  c_THHalfTensor_geev :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_gesvd"
  c_THHalfTensor_gesvd :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_gesvd2"
  c_THHalfTensor_gesvd2 :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_getri"
  c_THHalfTensor_getri :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_potrf"
  c_THHalfTensor_potrf :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_potrs"
  c_THHalfTensor_potrs :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_potri"
  c_THHalfTensor_potri :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ()

-- |c_THHalfTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_qr"
  c_THHalfTensor_qr :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_geqrf"
  c_THHalfTensor_geqrf :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_orgqr"
  c_THHalfTensor_orgqr :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_ormqr"
  c_THHalfTensor_ormqr :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THHalfTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_pstrf"
  c_THHalfTensor_pstrf :: (Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> (Ptr CTHHalfTensor) -> Ptr CChar -> THHalf -> IO ()

-- |c_THHalfTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_btrifact"
  c_THHalfTensor_btrifact :: (Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHHalfTensor) -> IO ()

-- |c_THHalfTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THHalfTensor_btrisolve"
  c_THHalfTensor_btrisolve :: (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> IO ()