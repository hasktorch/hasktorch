{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorLapack where

import Foreign
import Foreign.C.Types
import Data.Word
import Data.Int
import Torch.Types.TH

-- | c_gesv :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesv"
  c_gesv_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_gesv_ with unused argument (for CTHState) to unify backpack signatures.
c_gesv :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_gesv = const c_gesv_

-- | c_trtrs :  rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h THFloatTensor_trtrs"
  c_trtrs_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
c_trtrs :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()
c_trtrs = const c_trtrs_

-- | c_gels :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gels"
  c_gels_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_gels_ with unused argument (for CTHState) to unify backpack signatures.
c_gels :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_gels = const c_gels_

-- | c_syev :  re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_syev"
  c_syev_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_syev_ with unused argument (for CTHState) to unify backpack signatures.
c_syev :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()
c_syev = const c_syev_

-- | c_geev :  re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h THFloatTensor_geev"
  c_geev_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_geev_ with unused argument (for CTHState) to unify backpack signatures.
c_geev :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_geev = const c_geev_

-- | c_gesvd :  ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesvd"
  c_gesvd_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_gesvd = const c_gesvd_

-- | c_gesvd2 :  ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h THFloatTensor_gesvd2"
  c_gesvd2_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_gesvd2_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd2 :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_gesvd2 = const c_gesvd2_

-- | c_getri :  ra_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_getri"
  c_getri_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_getri_ with unused argument (for CTHState) to unify backpack signatures.
c_getri :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_getri = const c_getri_

-- | c_potrf :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potrf"
  c_potrf_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_potrf_ with unused argument (for CTHState) to unify backpack signatures.
c_potrf :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_potrf = const c_potrf_

-- | c_potrs :  rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potrs"
  c_potrs_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_potrs_ with unused argument (for CTHState) to unify backpack signatures.
c_potrs :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_potrs = const c_potrs_

-- | c_potri :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THFloatTensor_potri"
  c_potri_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()

-- | alias of c_potri_ with unused argument (for CTHState) to unify backpack signatures.
c_potri :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ()
c_potri = const c_potri_

-- | c_qr :  rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_qr"
  c_qr_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_qr_ with unused argument (for CTHState) to unify backpack signatures.
c_qr :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_qr = const c_qr_

-- | c_geqrf :  ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_geqrf"
  c_geqrf_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
c_geqrf :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_geqrf = const c_geqrf_

-- | c_orgqr :  ra_ a tau -> void
foreign import ccall "THTensorLapack.h THFloatTensor_orgqr"
  c_orgqr_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
c_orgqr :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ()
c_orgqr = const c_orgqr_

-- | c_ormqr :  ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h THFloatTensor_ormqr"
  c_ormqr_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
c_ormqr :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ()
c_ormqr = const c_ormqr_

-- | c_pstrf :  ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h THFloatTensor_pstrf"
  c_pstrf_ :: Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THFloatTensor -> Ptr CChar -> CFloat -> IO ()

-- | alias of c_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
c_pstrf :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THFloatTensor -> Ptr CChar -> CFloat -> IO ()
c_pstrf = const c_pstrf_

-- | c_btrifact :  ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h THFloatTensor_btrifact"
  c_btrifact_ :: Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THFloatTensor -> IO ()

-- | alias of c_btrifact_ with unused argument (for CTHState) to unify backpack signatures.
c_btrifact :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THFloatTensor -> IO ()
c_btrifact = const c_btrifact_

-- | c_btrisolve :  rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h THFloatTensor_btrisolve"
  c_btrisolve_ :: Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_btrisolve_ with unused argument (for CTHState) to unify backpack signatures.
c_btrisolve :: Ptr C'THState -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ()
c_btrisolve = const c_btrisolve_

-- | p_gesv : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesv"
  p_gesv :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_trtrs : Pointer to function : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_trtrs"
  p_trtrs :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- | p_gels : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gels"
  p_gels :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_syev : Pointer to function : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_syev"
  p_syev :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | p_geev : Pointer to function : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_geev"
  p_geev :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd : Pointer to function : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesvd"
  p_gesvd :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_gesvd2 : Pointer to function : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_gesvd2"
  p_gesvd2 :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_getri : Pointer to function : ra_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_getri"
  p_getri :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_potrf : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potrf"
  p_potrf :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_potrs : Pointer to function : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potrs"
  p_potrs :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_potri : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_potri"
  p_potri :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> IO ())

-- | p_qr : Pointer to function : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_qr"
  p_qr :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_geqrf : Pointer to function : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_geqrf"
  p_geqrf :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_orgqr : Pointer to function : ra_ a tau -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_orgqr"
  p_orgqr :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> IO ())

-- | p_ormqr : Pointer to function : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_ormqr"
  p_ormqr :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | p_pstrf : Pointer to function : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_pstrf"
  p_pstrf :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THFloatTensor -> Ptr CChar -> CFloat -> IO ())

-- | p_btrifact : Pointer to function : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_btrifact"
  p_btrifact :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THFloatTensor -> IO ())

-- | p_btrisolve : Pointer to function : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h &THFloatTensor_btrisolve"
  p_btrisolve :: FunPtr (Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THFloatTensor -> Ptr C'THIntTensor -> IO ())