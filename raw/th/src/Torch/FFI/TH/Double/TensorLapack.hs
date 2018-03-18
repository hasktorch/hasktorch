{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Double.TensorLapack where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_gesv :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesv"
  c_gesv_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_gesv_ with unused argument (for CTHState) to unify backpack signatures.
c_gesv = const c_gesv_

-- | c_trtrs :  rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_trtrs"
  c_trtrs_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
c_trtrs = const c_trtrs_

-- | c_gels :  rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gels"
  c_gels_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_gels_ with unused argument (for CTHState) to unify backpack signatures.
c_gels = const c_gels_

-- | c_syev :  re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_syev"
  c_syev_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_syev_ with unused argument (for CTHState) to unify backpack signatures.
c_syev = const c_syev_

-- | c_geev :  re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_geev"
  c_geev_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_geev_ with unused argument (for CTHState) to unify backpack signatures.
c_geev = const c_geev_

-- | c_gesvd :  ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesvd"
  c_gesvd_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd = const c_gesvd_

-- | c_gesvd2 :  ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_gesvd2"
  c_gesvd2_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_gesvd2_ with unused argument (for CTHState) to unify backpack signatures.
c_gesvd2 = const c_gesvd2_

-- | c_getri :  ra_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_getri"
  c_getri_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_getri_ with unused argument (for CTHState) to unify backpack signatures.
c_getri = const c_getri_

-- | c_potrf :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potrf"
  c_potrf_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_potrf_ with unused argument (for CTHState) to unify backpack signatures.
c_potrf = const c_potrf_

-- | c_potrs :  rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potrs"
  c_potrs_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_potrs_ with unused argument (for CTHState) to unify backpack signatures.
c_potrs = const c_potrs_

-- | c_potri :  ra_ a uplo -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_potri"
  c_potri_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ()

-- | alias of c_potri_ with unused argument (for CTHState) to unify backpack signatures.
c_potri = const c_potri_

-- | c_qr :  rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_qr"
  c_qr_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_qr_ with unused argument (for CTHState) to unify backpack signatures.
c_qr = const c_qr_

-- | c_geqrf :  ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_geqrf"
  c_geqrf_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
c_geqrf = const c_geqrf_

-- | c_orgqr :  ra_ a tau -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_orgqr"
  c_orgqr_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
c_orgqr = const c_orgqr_

-- | c_ormqr :  ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_ormqr"
  c_ormqr_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> IO ()

-- | alias of c_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
c_ormqr = const c_ormqr_

-- | c_pstrf :  ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_pstrf"
  c_pstrf_ :: Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> CDouble -> IO ()

-- | alias of c_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
c_pstrf = const c_pstrf_

-- | c_btrifact :  ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_btrifact"
  c_btrifact_ :: Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THDoubleTensor -> IO ()

-- | alias of c_btrifact_ with unused argument (for CTHState) to unify backpack signatures.
c_btrifact = const c_btrifact_

-- | c_btrisolve :  rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h THDoubleTensor_btrisolve"
  c_btrisolve_ :: Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> IO ()

-- | alias of c_btrisolve_ with unused argument (for CTHState) to unify backpack signatures.
c_btrisolve = const c_btrisolve_

-- | p_gesv : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_gesv"
  p_gesv_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_gesv_ with unused argument (for CTHState) to unify backpack signatures.
p_gesv = const p_gesv_

-- | p_trtrs : Pointer to function : rb_ ra_ b_ a_ uplo trans diag -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_trtrs"
  p_trtrs_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())

-- | alias of p_trtrs_ with unused argument (for CTHState) to unify backpack signatures.
p_trtrs = const p_trtrs_

-- | p_gels : Pointer to function : rb_ ra_ b_ a_ -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_gels"
  p_gels_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_gels_ with unused argument (for CTHState) to unify backpack signatures.
p_gels = const p_gels_

-- | p_syev : Pointer to function : re_ rv_ a_ jobz uplo -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_syev"
  p_syev_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | alias of p_syev_ with unused argument (for CTHState) to unify backpack signatures.
p_syev = const p_syev_

-- | p_geev : Pointer to function : re_ rv_ a_ jobvr -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_geev"
  p_geev_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_geev_ with unused argument (for CTHState) to unify backpack signatures.
p_geev = const p_geev_

-- | p_gesvd : Pointer to function : ru_ rs_ rv_ a jobu -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_gesvd"
  p_gesvd_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_gesvd_ with unused argument (for CTHState) to unify backpack signatures.
p_gesvd = const p_gesvd_

-- | p_gesvd2 : Pointer to function : ru_ rs_ rv_ ra_ a jobu -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_gesvd2"
  p_gesvd2_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_gesvd2_ with unused argument (for CTHState) to unify backpack signatures.
p_gesvd2 = const p_gesvd2_

-- | p_getri : Pointer to function : ra_ a -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_getri"
  p_getri_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_getri_ with unused argument (for CTHState) to unify backpack signatures.
p_getri = const p_getri_

-- | p_potrf : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_potrf"
  p_potrf_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_potrf_ with unused argument (for CTHState) to unify backpack signatures.
p_potrf = const p_potrf_

-- | p_potrs : Pointer to function : rb_ b_ a_ uplo -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_potrs"
  p_potrs_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_potrs_ with unused argument (for CTHState) to unify backpack signatures.
p_potrs = const p_potrs_

-- | p_potri : Pointer to function : ra_ a uplo -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_potri"
  p_potri_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> IO ())

-- | alias of p_potri_ with unused argument (for CTHState) to unify backpack signatures.
p_potri = const p_potri_

-- | p_qr : Pointer to function : rq_ rr_ a -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_qr"
  p_qr_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_qr_ with unused argument (for CTHState) to unify backpack signatures.
p_qr = const p_qr_

-- | p_geqrf : Pointer to function : ra_ rtau_ a -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_geqrf"
  p_geqrf_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_geqrf_ with unused argument (for CTHState) to unify backpack signatures.
p_geqrf = const p_geqrf_

-- | p_orgqr : Pointer to function : ra_ a tau -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_orgqr"
  p_orgqr_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_orgqr_ with unused argument (for CTHState) to unify backpack signatures.
p_orgqr = const p_orgqr_

-- | p_ormqr : Pointer to function : ra_ a tau c side trans -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_ormqr"
  p_ormqr_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> Ptr CChar -> IO ())

-- | alias of p_ormqr_ with unused argument (for CTHState) to unify backpack signatures.
p_ormqr = const p_ormqr_

-- | p_pstrf : Pointer to function : ra_ rpiv_ a uplo tol -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_pstrf"
  p_pstrf_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> Ptr C'THDoubleTensor -> Ptr CChar -> CDouble -> IO ())

-- | alias of p_pstrf_ with unused argument (for CTHState) to unify backpack signatures.
p_pstrf = const p_pstrf_

-- | p_btrifact : Pointer to function : ra_ rpivots_ rinfo_ pivot a -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_btrifact"
  p_btrifact_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> Ptr C'THIntTensor -> CInt -> Ptr C'THDoubleTensor -> IO ())

-- | alias of p_btrifact_ with unused argument (for CTHState) to unify backpack signatures.
p_btrifact = const p_btrifact_

-- | p_btrisolve : Pointer to function : rb_ b atf pivots -> void
foreign import ccall "THTensorLapack.h &THDoubleTensor_btrisolve"
  p_btrisolve_ :: FunPtr (Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THDoubleTensor -> Ptr C'THIntTensor -> IO ())

-- | alias of p_btrisolve_ with unused argument (for CTHState) to unify backpack signatures.
p_btrisolve = const p_btrisolve_