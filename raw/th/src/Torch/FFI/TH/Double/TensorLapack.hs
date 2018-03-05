{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorLapack
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
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_gesv"
  c_gesv :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_trtrs :  rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_trtrs"
  c_trtrs :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> IO (())

-- | c_gels :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_gels"
  c_gels :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_syev :  re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_syev"
  c_syev :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (())

-- | c_geev :  re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_geev"
  c_geev :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_gesvd :  ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_gesvd"
  c_gesvd :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_gesvd2 :  ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_gesvd2"
  c_gesvd2 :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_getri :  ra_ a -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_getri"
  c_getri :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_potrf :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_potrf"
  c_potrf :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_potrs :  rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_potrs"
  c_potrs :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_potri :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_potri"
  c_potri :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (())

-- | c_qr :  rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_qr"
  c_qr :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_geqrf :  ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_geqrf"
  c_geqrf :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_orgqr :  ra_ a tau -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_orgqr"
  c_orgqr :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_ormqr :  ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_ormqr"
  c_ormqr :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (())

-- | c_pstrf :  ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_pstrf"
  c_pstrf :: Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> CDouble -> IO (())

-- | c_btrifact :  ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_btrifact"
  c_btrifact :: Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHDoubleTensor) -> IO (())

-- | c_btrisolve :  rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h c_THTensorDouble_btrisolve"
  c_btrisolve :: Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> IO (())

-- | p_gesv : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_gesv"
  p_gesv :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_trtrs : Pointer to function : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_trtrs"
  p_trtrs :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> Ptr (CChar) -> IO (()))

-- | p_gels : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_gels"
  p_gels :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_syev : Pointer to function : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_syev"
  p_syev :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (()))

-- | p_geev : Pointer to function : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_geev"
  p_geev :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_gesvd : Pointer to function : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_gesvd"
  p_gesvd :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_gesvd2 : Pointer to function : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_gesvd2"
  p_gesvd2 :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_getri : Pointer to function : ra_ a -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_getri"
  p_getri :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_potrf : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_potrf"
  p_potrf :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_potrs : Pointer to function : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_potrs"
  p_potrs :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_potri : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_potri"
  p_potri :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> IO (()))

-- | p_qr : Pointer to function : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_qr"
  p_qr :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_geqrf : Pointer to function : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_geqrf"
  p_geqrf :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_orgqr : Pointer to function : ra_ a tau -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_orgqr"
  p_orgqr :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_ormqr : Pointer to function : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_ormqr"
  p_ormqr :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> Ptr (CChar) -> IO (()))

-- | p_pstrf : Pointer to function : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_pstrf"
  p_pstrf :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CChar) -> CDouble -> IO (()))

-- | p_btrifact : Pointer to function : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_btrifact"
  p_btrifact :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> Ptr (CTHIntTensor) -> CInt -> Ptr (CTHDoubleTensor) -> IO (()))

-- | p_btrisolve : Pointer to function : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h &p_THTensorDouble_btrisolve"
  p_btrisolve :: FunPtr (Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHDoubleTensor) -> Ptr (CTHIntTensor) -> IO (()))