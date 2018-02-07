{-# LANGUAGE ForeignFunctionInterface #-}

module THShortTensorLapack
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
  c_gesv :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_trtrs : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h trtrs"
  c_trtrs :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_gels : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h gels"
  c_gels :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_syev : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h syev"
  c_syev :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_geev : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h geev"
  c_geev :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_gesvd : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd"
  c_gesvd :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_gesvd2 : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h gesvd2"
  c_gesvd2 :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_getri : ra_ a -> void
foreign import ccall "THTensorLapack.h getri"
  c_getri :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_potrf : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potrf"
  c_potrf :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_potrs : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h potrs"
  c_potrs :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_potri : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h potri"
  c_potri :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> IO ()

-- | c_qr : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h qr"
  c_qr :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_geqrf : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h geqrf"
  c_geqrf :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_orgqr : ra_ a tau -> void
foreign import ccall "THTensorLapack.h orgqr"
  c_orgqr :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> IO ()

-- | c_ormqr : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h ormqr"
  c_ormqr :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | c_pstrf : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h pstrf"
  c_pstrf :: Ptr CTHShortTensor -> Ptr CTHIntTensor -> Ptr CTHShortTensor -> Ptr CChar -> CShort -> IO ()

-- | c_btrifact : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h btrifact"
  c_btrifact :: Ptr CTHShortTensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr CTHShortTensor -> IO ()

-- | c_btrisolve : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h btrisolve"
  c_btrisolve :: Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHShortTensor -> Ptr CTHIntTensor -> IO ()


