module Torch.Class.Tensor.Math.Blas where

import Data.Void
import Torch.Class.Types

class TensorMathBlas t where
  addmv_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addmm_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addr_        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addbmm_      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  baddbmm_     :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  dot          :: t -> t -> IO (HsAccReal t)

  -- baddbmm_ :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  -- baddbmm_ t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_baddbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

-- type IntTensor = Void
-- 
-- -- class CPUTensorMathLapack t where
-- class GPUTensorMathBlas t where
--   btrifact :: t -> IntTensor -> IntTensor -> Int -> t -> io ()
--   btrisolve :: t -> t -> t -> IntTensor -> io ()


