module Torch.Undefined.Tensor.Math.Blas where

import Foreign
import Torch.Sig.Types
import Torch.Sig.Types.Global

c_dot     :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> IO CAccReal
c_dot     = undefined
c_addmv   :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ()
c_addmv   = undefined
c_addmm   :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ()
c_addmm   = undefined
c_addr    :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ()
c_addr    = undefined
c_addbmm  :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ()
c_addbmm  = undefined
c_baddbmm :: Ptr CState -> Ptr CTensor -> CReal -> Ptr CTensor -> CReal -> Ptr CTensor -> Ptr CTensor -> IO ()
c_baddbmm = undefined

-- * THC Float Blas, TH Float Lapack
-- c_btrifact :: Ptr CState -> Ptr CTensor -> Ptr CIntTensor -> Ptr CIntTensor -> CInt -> Ptr CTensor -> IO ()
-- c_btrisolve :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> Ptr CTensor -> Ptr CIntTensor -> IO ()


