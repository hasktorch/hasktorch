{-# LANGUAGE TypeSynonymInstances #-}
module Torch.Core.Tensor.Generic.Lapack where

import Torch.Core.Tensor.Generic.Internal

import qualified THDoubleTensorLapack as T
import qualified THFloatTensorLapack as T

class LapackOps t where
    c_gesv      :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> IO ()
    c_trtrs     :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()
    c_gels      :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> IO ()
    c_syev      :: Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> IO ()
    c_geev      :: Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_gesvd     :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_gesvd2    :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_getri     :: Ptr t -> Ptr t -> IO ()
    c_potrf     :: Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_potrs     :: Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_potri     :: Ptr t -> Ptr t -> Ptr CChar -> IO ()
    c_qr        :: Ptr t -> Ptr t -> Ptr t -> IO ()
    c_geqrf     :: Ptr t -> Ptr t -> Ptr t -> IO ()
    c_orgqr     :: Ptr t -> Ptr t -> Ptr t -> IO ()
    c_ormqr     :: Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> IO ()
    c_pstrf     :: Ptr t -> Ptr CTHIntTensor -> Ptr t -> Ptr CChar -> HaskReal t -> IO ()
    c_btrifact  :: Ptr t -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr t -> IO ()
    c_btrisolve :: Ptr t -> Ptr t -> Ptr t -> Ptr CTHIntTensor -> IO ()
    p_gesv      :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> IO ())
    p_trtrs     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ())
    p_gels      :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> IO ())
    p_syev      :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> IO ())
    p_geev      :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_gesvd     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_gesvd2    :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_getri     :: FunPtr (Ptr t -> Ptr t -> IO ())
    p_potrf     :: FunPtr (Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_potrs     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_potri     :: FunPtr (Ptr t -> Ptr t -> Ptr CChar -> IO ())
    p_qr        :: FunPtr (Ptr t -> Ptr t -> Ptr t -> IO ())
    p_geqrf     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> IO ())
    p_orgqr     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> IO ())
    p_ormqr     :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr t -> Ptr CChar -> Ptr CChar -> IO ())
    p_pstrf     :: FunPtr (Ptr t -> Ptr CTHIntTensor -> Ptr t -> Ptr CChar -> HaskReal t -> IO ())
    p_btrifact  :: FunPtr (Ptr t -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr t -> IO ())
    p_btrisolve :: FunPtr (Ptr t -> Ptr t -> Ptr t -> Ptr CTHIntTensor -> IO ())

instance LapackOps CTHDoubleTensor where
    c_gesv      = T.c_THDoubleTensor_gesv
    c_trtrs     = T.c_THDoubleTensor_trtrs
    c_gels      = T.c_THDoubleTensor_gels
    c_syev      = T.c_THDoubleTensor_syev
    c_geev      = T.c_THDoubleTensor_geev
    c_gesvd     = T.c_THDoubleTensor_gesvd
    c_gesvd2    = T.c_THDoubleTensor_gesvd2
    c_getri     = T.c_THDoubleTensor_getri
    c_potrf     = T.c_THDoubleTensor_potrf
    c_potrs     = T.c_THDoubleTensor_potrs
    c_potri     = T.c_THDoubleTensor_potri
    c_qr        = T.c_THDoubleTensor_qr
    c_geqrf     = T.c_THDoubleTensor_geqrf
    c_orgqr     = T.c_THDoubleTensor_orgqr
    c_ormqr     = T.c_THDoubleTensor_ormqr
    c_pstrf     = T.c_THDoubleTensor_pstrf
    c_btrifact  = T.c_THDoubleTensor_btrifact
    c_btrisolve = T.c_THDoubleTensor_btrisolve
    p_gesv      = T.p_THDoubleTensor_gesv
    p_trtrs     = T.p_THDoubleTensor_trtrs
    p_gels      = T.p_THDoubleTensor_gels
    p_syev      = T.p_THDoubleTensor_syev
    p_geev      = T.p_THDoubleTensor_geev
    p_gesvd     = T.p_THDoubleTensor_gesvd
    p_gesvd2    = T.p_THDoubleTensor_gesvd2
    p_getri     = T.p_THDoubleTensor_getri
    p_potrf     = T.p_THDoubleTensor_potrf
    p_potrs     = T.p_THDoubleTensor_potrs
    p_potri     = T.p_THDoubleTensor_potri
    p_qr        = T.p_THDoubleTensor_qr
    p_geqrf     = T.p_THDoubleTensor_geqrf
    p_orgqr     = T.p_THDoubleTensor_orgqr
    p_ormqr     = T.p_THDoubleTensor_ormqr
    p_pstrf     = T.p_THDoubleTensor_pstrf
    p_btrifact  = T.p_THDoubleTensor_btrifact
    p_btrisolve = T.p_THDoubleTensor_btrisolve


instance LapackOps CTHFloatTensor where
    c_gesv      = T.c_THFloatTensor_gesv
    c_trtrs     = T.c_THFloatTensor_trtrs
    c_gels      = T.c_THFloatTensor_gels
    c_syev      = T.c_THFloatTensor_syev
    c_geev      = T.c_THFloatTensor_geev
    c_gesvd     = T.c_THFloatTensor_gesvd
    c_gesvd2    = T.c_THFloatTensor_gesvd2
    c_getri     = T.c_THFloatTensor_getri
    c_potrf     = T.c_THFloatTensor_potrf
    c_potrs     = T.c_THFloatTensor_potrs
    c_potri     = T.c_THFloatTensor_potri
    c_qr        = T.c_THFloatTensor_qr
    c_geqrf     = T.c_THFloatTensor_geqrf
    c_orgqr     = T.c_THFloatTensor_orgqr
    c_ormqr     = T.c_THFloatTensor_ormqr
    c_pstrf     = T.c_THFloatTensor_pstrf
    c_btrifact  = T.c_THFloatTensor_btrifact
    c_btrisolve = T.c_THFloatTensor_btrisolve
    p_gesv      = T.p_THFloatTensor_gesv
    p_trtrs     = T.p_THFloatTensor_trtrs
    p_gels      = T.p_THFloatTensor_gels
    p_syev      = T.p_THFloatTensor_syev
    p_geev      = T.p_THFloatTensor_geev
    p_gesvd     = T.p_THFloatTensor_gesvd
    p_gesvd2    = T.p_THFloatTensor_gesvd2
    p_getri     = T.p_THFloatTensor_getri
    p_potrf     = T.p_THFloatTensor_potrf
    p_potrs     = T.p_THFloatTensor_potrs
    p_potri     = T.p_THFloatTensor_potri
    p_qr        = T.p_THFloatTensor_qr
    p_geqrf     = T.p_THFloatTensor_geqrf
    p_orgqr     = T.p_THFloatTensor_orgqr
    p_ormqr     = T.p_THFloatTensor_ormqr
    p_pstrf     = T.p_THFloatTensor_pstrf
    p_btrifact  = T.p_THFloatTensor_btrifact
    p_btrisolve = T.p_THFloatTensor_btrisolve


