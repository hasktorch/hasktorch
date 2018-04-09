{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Blas where

import Data.Void
import Torch.Class.Types
import System.IO.Unsafe

class TensorMathBlas t where
  _addmv       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  _addmm       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  _addr        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  _addbmm      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  _baddbmm     :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  dot          :: t -> t -> IO (HsAccReal t)

  -- _baddbmm :: Dynamic -> HsReal -> Dynamic -> HsReal -> Dynamic -> Dynamic -> IO ()
  -- _baddbmm t0 v0 t1 v1 t2 t3 = _with4Tensors t0 t1 t2 t3 $ \t0' t1' t2' t3' -> Sig.c_baddbmm t0' (hs2cReal v0) t1' (hs2cReal v1) t2' t3'

-- type IntTensor = Void
-- 
-- -- class CPUTensorMathLapack t where
-- class GPUTensorMathBlas t where
--   btrifact :: t -> IntTensor -> IntTensor -> Int -> t -> io ()
--   btrisolve :: t -> t -> t -> IntTensor -> io ()


(<.>) :: TensorMathBlas t => t -> t -> HsAccReal t
(<.>) a b = unsafePerformIO $ dot a b
{-# NOINLINE (<.>) #-}

