module Torch.Undefined.Tensor.Math.Reduce.Floating where

import Foreign
import Foreign.C.Types
import Torch.Sig.Types
import Torch.Sig.Types.Global

c_dist :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CReal -> IO CAccReal
c_dist = undefined
c_var :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
c_var = undefined
c_varall :: Ptr CState -> Ptr CTensor -> CInt -> IO CAccReal
c_varall = undefined
c_std :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> CInt -> IO ()
c_std = undefined
c_stdall :: Ptr CState -> Ptr CTensor -> CInt -> IO CAccReal
c_stdall = undefined
c_renorm :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CReal -> CInt -> CReal -> IO ()
c_renorm = undefined
c_norm :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CReal -> CInt -> CInt -> IO ()
c_norm = undefined
c_normall :: Ptr CState -> Ptr CTensor -> CReal -> IO CAccReal
c_normall = undefined
c_mean  :: Ptr CState -> Ptr CTensor -> Ptr CTensor -> CInt -> CInt -> IO ()
c_mean  = undefined
c_meanall :: Ptr CState -> Ptr CTensor -> IO CAccReal
c_meanall = undefined


