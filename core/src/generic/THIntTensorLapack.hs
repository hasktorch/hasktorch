{-# LANGUAGE ForeignFunctionInterface #-}

module THIntTensorLapack (
    c_THIntTensor_gesv,
    c_THIntTensor_trtrs,
    c_THIntTensor_gels,
    c_THIntTensor_syev,
    c_THIntTensor_geev,
    c_THIntTensor_gesvd,
    c_THIntTensor_gesvd2,
    c_THIntTensor_getri,
    c_THIntTensor_potrf,
    c_THIntTensor_potrs,
    c_THIntTensor_potri,
    c_THIntTensor_qr,
    c_THIntTensor_geqrf,
    c_THIntTensor_orgqr,
    c_THIntTensor_ormqr,
    c_THIntTensor_pstrf,
    c_THIntTensor_btrifact,
    c_THIntTensor_btrisolve,
    p_THIntTensor_gesv,
    p_THIntTensor_trtrs,
    p_THIntTensor_gels,
    p_THIntTensor_syev,
    p_THIntTensor_geev,
    p_THIntTensor_gesvd,
    p_THIntTensor_gesvd2,
    p_THIntTensor_getri,
    p_THIntTensor_potrf,
    p_THIntTensor_potrs,
    p_THIntTensor_potri,
    p_THIntTensor_qr,
    p_THIntTensor_geqrf,
    p_THIntTensor_orgqr,
    p_THIntTensor_ormqr,
    p_THIntTensor_pstrf,
    p_THIntTensor_btrifact,
    p_THIntTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THIntTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_gesv"
  c_THIntTensor_gesv :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_trtrs"
  c_THIntTensor_trtrs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_gels"
  c_THIntTensor_gels :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_syev"
  c_THIntTensor_syev :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_geev"
  c_THIntTensor_geev :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_gesvd"
  c_THIntTensor_gesvd :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_gesvd2"
  c_THIntTensor_gesvd2 :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_getri"
  c_THIntTensor_getri :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_potrf"
  c_THIntTensor_potrf :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_potrs"
  c_THIntTensor_potrs :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_potri"
  c_THIntTensor_potri :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ()

-- |c_THIntTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_qr"
  c_THIntTensor_qr :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_geqrf"
  c_THIntTensor_geqrf :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_orgqr"
  c_THIntTensor_orgqr :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_ormqr"
  c_THIntTensor_ormqr :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THIntTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_pstrf"
  c_THIntTensor_pstrf :: (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> (Ptr CTHIntTensor) -> Ptr CChar -> CInt -> IO ()

-- |c_THIntTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_btrifact"
  c_THIntTensor_btrifact :: (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHIntTensor) -> IO ()

-- |c_THIntTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THIntTensor_btrisolve"
  c_THIntTensor_btrisolve :: (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> IO ()

-- |p_THIntTensor_gesv : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_gesv"
  p_THIntTensor_gesv :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_trtrs : Pointer to rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_trtrs"
  p_THIntTensor_trtrs :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THIntTensor_gels : Pointer to rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_gels"
  p_THIntTensor_gels :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_syev : Pointer to re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_syev"
  p_THIntTensor_syev :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THIntTensor_geev : Pointer to re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_geev"
  p_THIntTensor_geev :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_gesvd : Pointer to ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_gesvd"
  p_THIntTensor_gesvd :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_gesvd2 : Pointer to ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_gesvd2"
  p_THIntTensor_gesvd2 :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_getri : Pointer to ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_getri"
  p_THIntTensor_getri :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_potrf : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_potrf"
  p_THIntTensor_potrf :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_potrs : Pointer to rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_potrs"
  p_THIntTensor_potrs :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_potri : Pointer to ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_potri"
  p_THIntTensor_potri :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> IO ())

-- |p_THIntTensor_qr : Pointer to rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_qr"
  p_THIntTensor_qr :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_geqrf : Pointer to ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_geqrf"
  p_THIntTensor_geqrf :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_orgqr : Pointer to ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_orgqr"
  p_THIntTensor_orgqr :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_ormqr : Pointer to ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_ormqr"
  p_THIntTensor_ormqr :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THIntTensor_pstrf : Pointer to ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_pstrf"
  p_THIntTensor_pstrf :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntTensor -> (Ptr CTHIntTensor) -> Ptr CChar -> CInt -> IO ())

-- |p_THIntTensor_btrifact : Pointer to ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_btrifact"
  p_THIntTensor_btrifact :: FunPtr ((Ptr CTHIntTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHIntTensor) -> IO ())

-- |p_THIntTensor_btrisolve : Pointer to rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h &THIntTensor_btrisolve"
  p_THIntTensor_btrisolve :: FunPtr ((Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> (Ptr CTHIntTensor) -> Ptr CTHIntTensor -> IO ())