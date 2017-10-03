{-# LANGUAGE ForeignFunctionInterface #-}

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
    c_THDoubleTensor_btrisolve,
    p_THDoubleTensor_gesv,
    p_THDoubleTensor_trtrs,
    p_THDoubleTensor_gels,
    p_THDoubleTensor_syev,
    p_THDoubleTensor_geev,
    p_THDoubleTensor_gesvd,
    p_THDoubleTensor_gesvd2,
    p_THDoubleTensor_getri,
    p_THDoubleTensor_potrf,
    p_THDoubleTensor_potrs,
    p_THDoubleTensor_potri,
    p_THDoubleTensor_qr,
    p_THDoubleTensor_geqrf,
    p_THDoubleTensor_orgqr,
    p_THDoubleTensor_ormqr,
    p_THDoubleTensor_pstrf,
    p_THDoubleTensor_btrifact,
    p_THDoubleTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THDoubleTensor_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_gesv"
  c_THDoubleTensor_gesv :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_trtrs"
  c_THDoubleTensor_trtrs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_gels : rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_gels"
  c_THDoubleTensor_gels :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_syev"
  c_THDoubleTensor_syev :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_geev : re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_geev"
  c_THDoubleTensor_geev :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_gesvd"
  c_THDoubleTensor_gesvd :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_gesvd2"
  c_THDoubleTensor_gesvd2 :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_getri : ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_getri"
  c_THDoubleTensor_getri :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_potrf : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_potrf"
  c_THDoubleTensor_potrf :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_potrs : rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_potrs"
  c_THDoubleTensor_potrs :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_potri : ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_potri"
  c_THDoubleTensor_potri :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_qr : rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_qr"
  c_THDoubleTensor_qr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_geqrf : ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_geqrf"
  c_THDoubleTensor_geqrf :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_orgqr : ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_orgqr"
  c_THDoubleTensor_orgqr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_ormqr : ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_ormqr"
  c_THDoubleTensor_ormqr :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> IO ()

-- |c_THDoubleTensor_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_pstrf"
  c_THDoubleTensor_pstrf :: (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> (Ptr CTHDoubleTensor) -> Ptr CChar -> CDouble -> IO ()

-- |c_THDoubleTensor_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_btrifact"
  c_THDoubleTensor_btrifact :: (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHDoubleTensor) -> IO ()

-- |c_THDoubleTensor_btrisolve : rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h THDoubleTensor_btrisolve"
  c_THDoubleTensor_btrisolve :: (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> IO ()

-- |p_THDoubleTensor_gesv : Pointer to function rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_gesv"
  p_THDoubleTensor_gesv :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_trtrs : Pointer to function rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_trtrs"
  p_THDoubleTensor_trtrs :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_gels : Pointer to function rb_ ra_ b_ a_ -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_gels"
  p_THDoubleTensor_gels :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_syev : Pointer to function re_ rv_ a_ jobz uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_syev"
  p_THDoubleTensor_syev :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_geev : Pointer to function re_ rv_ a_ jobvr -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_geev"
  p_THDoubleTensor_geev :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_gesvd : Pointer to function ru_ rs_ rv_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_gesvd"
  p_THDoubleTensor_gesvd :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_gesvd2 : Pointer to function ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_gesvd2"
  p_THDoubleTensor_gesvd2 :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_getri : Pointer to function ra_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_getri"
  p_THDoubleTensor_getri :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_potrf : Pointer to function ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_potrf"
  p_THDoubleTensor_potrf :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_potrs : Pointer to function rb_ b_ a_ uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_potrs"
  p_THDoubleTensor_potrs :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_potri : Pointer to function ra_ a uplo -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_potri"
  p_THDoubleTensor_potri :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_qr : Pointer to function rq_ rr_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_qr"
  p_THDoubleTensor_qr :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_geqrf : Pointer to function ra_ rtau_ a -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_geqrf"
  p_THDoubleTensor_geqrf :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_orgqr : Pointer to function ra_ a tau -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_orgqr"
  p_THDoubleTensor_orgqr :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_ormqr : Pointer to function ra_ a tau c side trans -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_ormqr"
  p_THDoubleTensor_ormqr :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THDoubleTensor_pstrf : Pointer to function ra_ rpiv_ a uplo tol -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_pstrf"
  p_THDoubleTensor_pstrf :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> (Ptr CTHDoubleTensor) -> Ptr CChar -> CDouble -> IO ())

-- |p_THDoubleTensor_btrifact : Pointer to function ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_btrifact"
  p_THDoubleTensor_btrifact :: FunPtr ((Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHDoubleTensor) -> IO ())

-- |p_THDoubleTensor_btrisolve : Pointer to function rb_ b atf pivots -> void
foreign import ccall unsafe "THTensorLapack.h &THDoubleTensor_btrisolve"
  p_THDoubleTensor_btrisolve :: FunPtr ((Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> (Ptr CTHDoubleTensor) -> Ptr CTHIntTensor -> IO ())