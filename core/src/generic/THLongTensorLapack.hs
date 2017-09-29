{-# LANGUAGE ForeignFunctionInterface #-}

module THLongTensorLapack (
    c_THLongTensor_gesv,
    c_THLongTensor_trtrs,
    c_THLongTensor_gels,
    c_THLongTensor_syev,
    c_THLongTensor_geev,
    c_THLongTensor_gesvd,
    c_THLongTensor_gesvd2,
    c_THLongTensor_getri,
    c_THLongTensor_potrf,
    c_THLongTensor_potrs,
    c_THLongTensor_potri,
    c_THLongTensor_qr,
    c_THLongTensor_geqrf,
    c_THLongTensor_orgqr,
    c_THLongTensor_ormqr,
    c_THLongTensor_pstrf,
    c_THLongTensor_btrifact,
    c_THLongTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THLongTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_gesv"
  c_THLongTensor_gesv :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_trtrs"
  c_THLongTensor_trtrs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_gels"
  c_THLongTensor_gels :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_syev"
  c_THLongTensor_syev :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_geev"
  c_THLongTensor_geev :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_gesvd"
  c_THLongTensor_gesvd :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_gesvd2"
  c_THLongTensor_gesvd2 :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_getri"
  c_THLongTensor_getri :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_potrf"
  c_THLongTensor_potrf :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_potrs"
  c_THLongTensor_potrs :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_potri"
  c_THLongTensor_potri :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ()

-- |c_THLongTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_qr"
  c_THLongTensor_qr :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_geqrf"
  c_THLongTensor_geqrf :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_orgqr"
  c_THLongTensor_orgqr :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_ormqr"
  c_THLongTensor_ormqr :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THLongTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_pstrf"
  c_THLongTensor_pstrf :: (Ptr CTHLongTensor) -> Ptr CTHIntTensor -> (Ptr CTHLongTensor) -> Ptr CChar -> CLong -> IO ()

-- |c_THLongTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_btrifact"
  c_THLongTensor_btrifact :: (Ptr CTHLongTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHLongTensor) -> IO ()

-- |c_THLongTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THLongTensor_btrisolve"
  c_THLongTensor_btrisolve :: (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHIntTensor -> IO ()