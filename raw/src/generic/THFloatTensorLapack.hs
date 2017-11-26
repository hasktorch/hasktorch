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
    c_THFloatTensor_btrisolve,
    p_THFloatTensor_gesv,
    p_THFloatTensor_trtrs,
    p_THFloatTensor_gels,
    p_THFloatTensor_syev,
    p_THFloatTensor_geev,
    p_THFloatTensor_gesvd,
    p_THFloatTensor_gesvd2,
    p_THFloatTensor_getri,
    p_THFloatTensor_potrf,
    p_THFloatTensor_potrs,
    p_THFloatTensor_potri,
    p_THFloatTensor_qr,
    p_THFloatTensor_geqrf,
    p_THFloatTensor_orgqr,
    p_THFloatTensor_ormqr,
    p_THFloatTensor_pstrf,
    p_THFloatTensor_btrifact,
    p_THFloatTensor_btrisolve) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

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

-- |p_THFloatTensor_gesv : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesv"
  p_THFloatTensor_gesv :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_trtrs : Pointer to function : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_trtrs"
  p_THFloatTensor_trtrs :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_gels : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gels"
  p_THFloatTensor_gels :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_syev : Pointer to function : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_syev"
  p_THFloatTensor_syev :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_geev : Pointer to function : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_geev"
  p_THFloatTensor_geev :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_gesvd : Pointer to function : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesvd"
  p_THFloatTensor_gesvd :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_gesvd2 : Pointer to function : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesvd2"
  p_THFloatTensor_gesvd2 :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_getri : Pointer to function : ra_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_getri"
  p_THFloatTensor_getri :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_potrf : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potrf"
  p_THFloatTensor_potrf :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_potrs : Pointer to function : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potrs"
  p_THFloatTensor_potrs :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_potri : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potri"
  p_THFloatTensor_potri :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> IO ())

-- |p_THFloatTensor_qr : Pointer to function : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_qr"
  p_THFloatTensor_qr :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_geqrf : Pointer to function : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_geqrf"
  p_THFloatTensor_geqrf :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_orgqr : Pointer to function : ra_ a tau -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_orgqr"
  p_THFloatTensor_orgqr :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_ormqr : Pointer to function : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_ormqr"
  p_THFloatTensor_ormqr :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_THFloatTensor_pstrf : Pointer to function : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_pstrf"
  p_THFloatTensor_pstrf :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> (Ptr CTHFloatTensor) -> Ptr CChar -> CFloat -> IO ())

-- |p_THFloatTensor_btrifact : Pointer to function : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_btrifact"
  p_THFloatTensor_btrifact :: FunPtr ((Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> (Ptr CTHFloatTensor) -> IO ())

-- |p_THFloatTensor_btrisolve : Pointer to function : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_btrisolve"
  p_THFloatTensor_btrisolve :: FunPtr ((Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> Ptr CTHIntTensor -> IO ())