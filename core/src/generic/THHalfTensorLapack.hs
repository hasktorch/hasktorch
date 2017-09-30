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
    c_THHalfTensor_btrisolve,
    p_THHalfTensor_gesv,
    p_THHalfTensor_trtrs,
    p_THHalfTensor_gels,
    p_THHalfTensor_syev,
    p_THHalfTensor_geev,
    p_THHalfTensor_gesvd,
    p_THHalfTensor_gesvd2,
    p_THHalfTensor_getri,
    p_THHalfTensor_potrf,
    p_THHalfTensor_potrs,
    p_THHalfTensor_potri,
    p_THHalfTensor_qr,
    p_THHalfTensor_geqrf,
    p_THHalfTensor_orgqr,
    p_THHalfTensor_ormqr,
    p_THHalfTensor_pstrf,
    p_THHalfTensor_btrifact,
    p_THHalfTensor_btrisolve) where

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

-- |p_THHalfTensor_gesv : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_gesv"
  p_THHalfTensor_gesv :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_trtrs : Pointer to rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_trtrs"
  p_THHalfTensor_trtrs :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_gels : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_gels"
  p_THHalfTensor_gels :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_syev : Pointer to re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_syev"
  p_THHalfTensor_syev :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_geev : Pointer to re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_geev"
  p_THHalfTensor_geev :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_gesvd : Pointer to ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_gesvd"
  p_THHalfTensor_gesvd :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_gesvd2 : Pointer to ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_gesvd2"
  p_THHalfTensor_gesvd2 :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_getri : Pointer to ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_getri"
  p_THHalfTensor_getri :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_potrf : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_potrf"
  p_THHalfTensor_potrf :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_potrs : Pointer to rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_potrs"
  p_THHalfTensor_potrs :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_potri : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_potri"
  p_THHalfTensor_potri :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> IO ())

-- |p_THHalfTensor_qr : Pointer to rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_qr"
  p_THHalfTensor_qr :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_geqrf : Pointer to ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_geqrf"
  p_THHalfTensor_geqrf :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_orgqr : Pointer to ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_orgqr"
  p_THHalfTensor_orgqr :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_ormqr : Pointer to ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_ormqr"
  p_THHalfTensor_ormqr :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THHalfTensor_pstrf : Pointer to ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_pstrf"
  p_THHalfTensor_pstrf :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> (Ptr CTHHalfTensor) -> Ptr CChar -> THHalf -> IO ())

-- |p_THHalfTensor_btrifact : Pointer to ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_btrifact"
  p_THHalfTensor_btrifact :: FunPtr ((Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHHalfTensor) -> IO ())

-- |p_THHalfTensor_btrisolve : Pointer to rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h &THHalfTensor_btrisolve"
  p_THHalfTensor_btrisolve :: FunPtr ((Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> (Ptr CTHHalfTensor) -> Ptr CTHIntTensor -> IO ())