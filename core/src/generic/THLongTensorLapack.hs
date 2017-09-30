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
    c_THLongTensor_btrisolve,
    p_THLongTensor_gesv,
    p_THLongTensor_trtrs,
    p_THLongTensor_gels,
    p_THLongTensor_syev,
    p_THLongTensor_geev,
    p_THLongTensor_gesvd,
    p_THLongTensor_gesvd2,
    p_THLongTensor_getri,
    p_THLongTensor_potrf,
    p_THLongTensor_potrs,
    p_THLongTensor_potri,
    p_THLongTensor_qr,
    p_THLongTensor_geqrf,
    p_THLongTensor_orgqr,
    p_THLongTensor_ormqr,
    p_THLongTensor_pstrf,
    p_THLongTensor_btrifact,
    p_THLongTensor_btrisolve) where

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

-- |p_THLongTensor_gesv : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_gesv"
  p_THLongTensor_gesv :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_trtrs : Pointer to rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_trtrs"
  p_THLongTensor_trtrs :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_gels : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_gels"
  p_THLongTensor_gels :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_syev : Pointer to re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_syev"
  p_THLongTensor_syev :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_geev : Pointer to re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_geev"
  p_THLongTensor_geev :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_gesvd : Pointer to ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_gesvd"
  p_THLongTensor_gesvd :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_gesvd2 : Pointer to ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_gesvd2"
  p_THLongTensor_gesvd2 :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_getri : Pointer to ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_getri"
  p_THLongTensor_getri :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_potrf : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_potrf"
  p_THLongTensor_potrf :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_potrs : Pointer to rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_potrs"
  p_THLongTensor_potrs :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_potri : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_potri"
  p_THLongTensor_potri :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> IO ())

-- |p_THLongTensor_qr : Pointer to rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_qr"
  p_THLongTensor_qr :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_geqrf : Pointer to ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_geqrf"
  p_THLongTensor_geqrf :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_orgqr : Pointer to ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_orgqr"
  p_THLongTensor_orgqr :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_ormqr : Pointer to ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_ormqr"
  p_THLongTensor_ormqr :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THLongTensor_pstrf : Pointer to ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_pstrf"
  p_THLongTensor_pstrf :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHIntTensor -> (Ptr CTHLongTensor) -> Ptr CChar -> CLong -> IO ())

-- |p_THLongTensor_btrifact : Pointer to ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_btrifact"
  p_THLongTensor_btrifact :: FunPtr ((Ptr CTHLongTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHLongTensor) -> IO ())

-- |p_THLongTensor_btrisolve : Pointer to rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h &THLongTensor_btrisolve"
  p_THLongTensor_btrisolve :: FunPtr ((Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> (Ptr CTHLongTensor) -> Ptr CTHIntTensor -> IO ())