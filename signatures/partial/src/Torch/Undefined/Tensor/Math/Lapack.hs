module Torch.Undefined.Tensor.Math.Lapack where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global

c_gesv :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_gesv = undefined
c_gels :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_gels = undefined
c_syev :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> Ptr CChar -> IO ()
c_syev = undefined
c_geev :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_geev = undefined
c_gesvd :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_gesvd = undefined
c_gesvd2 :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_gesvd2 = undefined
c_getri :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO ()
c_getri = undefined
c_potri :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_potri = undefined
c_potrf :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_potrf = undefined
c_potrs :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> IO ()
c_potrs = undefined
c_geqrf :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_geqrf = undefined
c_qr :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
c_qr = undefined

-- * TH
-- c_trtrs     :: Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> Ptr CChar -> Ptr CChar -> IO ()
-- c_orgqr     :: Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> IO ()
-- c_ormqr     :: Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CChar -> Ptr CChar -> IO ()
-- c_pstrf     :: Ptr CTensor -> Ptr CTHIntTensor -> Ptr CTensor -> Ptr CChar -> CReal -> IO ()
-- c_btrifact  :: Ptr CTensor -> Ptr CTHIntTensor -> Ptr CTHIntTensor -> CInt -> Ptr CTensor -> IO ()
-- c_btrisolve :: Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CTHIntTensor -> IO ()
