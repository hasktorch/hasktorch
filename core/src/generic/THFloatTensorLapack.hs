{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorLapack (
    c_THFloatTensor_gesv,
    c_THFloatTensor_trtrs,
    c_THFloatTensor_gels,
    c_THFloatTensor_syev,
    c_THFloatTensor_geev,
    c_THFloatTensor_gesvd,
    c_THFloatTensor_gesvd2,
    c_THFloatTensor_getri,
    c_THFloatTensor_potrf,
    c_THFloatTensor_potrs,
    c_THFloatTensor_potri,
    c_THFloatTensor_qr,
    c_THFloatTensor_geqrf,
    c_THFloatTensor_orgqr,
    c_THFloatTensor_ormqr,
    c_THFloatTensor_pstrf,
    c_THFloatTensor_btrifact,
    c_THFloatTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THFloatTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesv"
  c_THFloatTensor_gesv :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h THFloatTensor_trtrs"
  c_THFloatTensor_trtrs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gels"
  c_THFloatTensor_gels :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_syev"
  c_THFloatTensor_syev :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h THFloatTensor_geev"
  c_THFloatTensor_geev :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesvd"
  c_THFloatTensor_gesvd :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesvd2"
  c_THFloatTensor_gesvd2 :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_getri : ra_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_getri"
  c_THFloatTensor_getri :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potrf"
  c_THFloatTensor_potrf :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potrs"
  c_THFloatTensor_potrs :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_potri : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potri"
  c_THFloatTensor_potri :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ()

-- |c_THFloatTensor_qr : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_qr"
  c_THFloatTensor_qr :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_geqrf"
  c_THFloatTensor_geqrf :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_orgqr : ra_ a tau -> void
foreign import ccall "THTensorLapack.h THFloatTensor_orgqr"
  c_THFloatTensor_orgqr :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h THFloatTensor_ormqr"
  c_THFloatTensor_ormqr :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THFloatTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h THFloatTensor_pstrf"
  c_THFloatTensor_pstrf :: (Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> (Ptr CTHFloatTensor) -> Ptr CChar -> CFloat -> IO ()

-- |c_THFloatTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_btrifact"
  c_THFloatTensor_btrifact :: (Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHFloatTensor) -> IO ()

-- |c_THFloatTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h THFloatTensor_btrisolve"
  c_THFloatTensor_btrisolve :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> IO ()