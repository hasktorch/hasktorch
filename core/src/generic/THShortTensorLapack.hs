{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorLapack (
    c_THShortTensor_gesv,
    c_THShortTensor_trtrs,
    c_THShortTensor_gels,
    c_THShortTensor_syev,
    c_THShortTensor_geev,
    c_THShortTensor_gesvd,
    c_THShortTensor_gesvd2,
    c_THShortTensor_getri,
    c_THShortTensor_potrf,
    c_THShortTensor_potrs,
    c_THShortTensor_potri,
    c_THShortTensor_qr,
    c_THShortTensor_geqrf,
    c_THShortTensor_orgqr,
    c_THShortTensor_ormqr,
    c_THShortTensor_pstrf,
    c_THShortTensor_btrifact,
    c_THShortTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_gesv"
  c_THShortTensor_gesv :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_trtrs"
  c_THShortTensor_trtrs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THShortTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_gels"
  c_THShortTensor_gels :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_syev"
  c_THShortTensor_syev :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THShortTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_geev"
  c_THShortTensor_geev :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_gesvd"
  c_THShortTensor_gesvd :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_gesvd2"
  c_THShortTensor_gesvd2 :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_getri"
  c_THShortTensor_getri :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_potrf"
  c_THShortTensor_potrf :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_potrs"
  c_THShortTensor_potrs :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_potri"
  c_THShortTensor_potri :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> IO ()

-- |c_THShortTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_qr"
  c_THShortTensor_qr :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_geqrf"
  c_THShortTensor_geqrf :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_orgqr"
  c_THShortTensor_orgqr :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_ormqr"
  c_THShortTensor_ormqr :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THShortTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_pstrf"
  c_THShortTensor_pstrf :: (Ptr CTHShortTensor) -> Ptr CTHIntTensor -> (Ptr CTHShortTensor) -> Ptr CChar -> CShort -> IO ()

-- |c_THShortTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_btrifact"
  c_THShortTensor_btrifact :: (Ptr CTHShortTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHShortTensor) -> IO ()

-- |c_THShortTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THShortTensor_btrisolve"
  c_THShortTensor_btrisolve :: (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> (Ptr CTHShortTensor) -> Ptr CTHIntTensor -> IO ()