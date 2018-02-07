{-# LANGUAGE ForeignFunctionInterface #-}

module THByteTensorLapack
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
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_gesv : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h gesv"
  c_gesv :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h trtrs"
  c_trtrs :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_gels : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h gels"
  c_gels :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h syev"
  c_syev :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_geev : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h geev"
  c_geev :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd"
  c_gesvd :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd2"
  c_gesvd2 :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_getri : ra_ a -> void
foreign import ccall "THTensorLapack.h getri"
  c_getri :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potrf"
  c_potrf :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_potrs : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h potrs"
  c_potrs :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_potri : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potri"
  c_potri :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> IO ()

-- | c_qr : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h qr"
  c_qr :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_geqrf : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h geqrf"
  c_geqrf :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_orgqr : ra_ a tau -> void
foreign import ccall "THTensorLapack.h orgqr"
  c_orgqr :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> IO ()

-- | c_ormqr : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h ormqr"
  c_ormqr :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h pstrf"
  c_pstrf :: Ptr CTHByteTensor -> Ptr CTHIntTensor -> Ptr CTHByteTensor -> Ptr CChar -> CChar -> IO ()

-- | c_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h btrifact"
  c_btrifact :: Ptr CTHByteTensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr CTHByteTensor -> IO ()

-- | c_btrisolve : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h btrisolve"
  c_btrisolve :: Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHByteTensor -> Ptr CTHIntTensor -> IO ()


