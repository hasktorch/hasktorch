{-# LANGUAGE ForeignFunctionInterface #-}

module THFloatTensorLapack
  ( c_gesv
  , c_trtrs
  , c_gels
  , c_syev
  , c_geev
  , c_gesvd
  , c_gesvd2
  , c_getri
  , c_potrf
  , c_potrs
  , c_potri
  , c_qr
  , c_geqrf
  , c_orgqr
  , c_ormqr
  , c_pstrf
  , c_btrifact
  , c_btrisolve
  , p_gesv
  , p_trtrs
  , p_gels
  , p_syev
  , p_geev
  , p_gesvd
  , p_gesvd2
  , p_getri
  , p_potrf
  , p_potrs
  , p_potri
  , p_qr
  , p_geqrf
  , p_orgqr
  , p_ormqr
  , p_pstrf
  , p_btrifact
  , p_btrisolve
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h gesv"
  c_gesv :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h trtrs"
  c_trtrs :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_gels : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h gels"
  c_gels :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h syev"
  c_syev :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_geev : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h geev"
  c_geev :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd"
  c_gesvd :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd2"
  c_gesvd2 :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_getri : ra_ a -> void
foreign import ccall "THTensorLapack.h getri"
  c_getri :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potrf"
  c_potrf :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_potrs : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h potrs"
  c_potrs :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_potri : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potri"
  c_potri :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ()

-- | c_qr : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h qr"
  c_qr :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_geqrf : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h geqrf"
  c_geqrf :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_orgqr : ra_ a tau -> void
foreign import ccall "THTensorLapack.h orgqr"
  c_orgqr :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ()

-- | c_ormqr : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h ormqr"
  c_ormqr :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h pstrf"
  c_pstrf :: Ptr CTHFloatTensor -> Ptr CTHIntTensor -> Ptr CTHFloatTensor -> Ptr CChar -> CFloat -> IO ()

-- | c_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h btrifact"
  c_btrifact :: Ptr CTHFloatTensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr CTHFloatTensor -> IO ()

-- | c_btrisolve : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h btrisolve"
  c_btrisolve :: Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHIntTensor -> IO ()

-- |p_gesv : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &gesv"
  p_gesv :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_trtrs : Pointer to function : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h &trtrs"
  p_trtrs :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_gels : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &gels"
  p_gels :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_syev : Pointer to function : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h &syev"
  p_syev :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_geev : Pointer to function : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h &geev"
  p_geev :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_gesvd : Pointer to function : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h &gesvd"
  p_gesvd :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_gesvd2 : Pointer to function : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h &gesvd2"
  p_gesvd2 :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_getri : Pointer to function : ra_ a -> void
foreign import ccall "THTensorLapack.h &getri"
  p_getri :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_potrf : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &potrf"
  p_potrf :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_potrs : Pointer to function : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h &potrs"
  p_potrs :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_potri : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &potri"
  p_potri :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> IO ())

-- |p_qr : Pointer to function : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h &qr"
  p_qr :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_geqrf : Pointer to function : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h &geqrf"
  p_geqrf :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_orgqr : Pointer to function : ra_ a tau -> void
foreign import ccall "THTensorLapack.h &orgqr"
  p_orgqr :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> IO ())

-- |p_ormqr : Pointer to function : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h &ormqr"
  p_ormqr :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- |p_pstrf : Pointer to function : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h &pstrf"
  p_pstrf :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHIntTensor -> Ptr CTHFloatTensor -> Ptr CChar -> CFloat -> IO ())

-- |p_btrifact : Pointer to function : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h &btrifact"
  p_btrifact :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr CTHFloatTensor -> IO ())

-- |p_btrisolve : Pointer to function : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h &btrisolve"
  p_btrisolve :: FunPtr (Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHFloatTensor -> Ptr CTHIntTensor -> IO ())