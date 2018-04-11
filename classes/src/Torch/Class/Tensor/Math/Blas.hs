{-# OPTIONS_GHC -fno-cse #-}
module Torch.Class.Tensor.Math.Blas where

import Data.Void
import Torch.Class.Types
import Torch.Class.Tensor
import System.IO.Unsafe

class IsTensor t => TensorMathBlas t where
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

mkInplaceFunction, mkNewFunction
  :: TensorMathBlas t
  => (t -> HsReal t -> t -> HsReal t -> t -> t -> IO ())
  -> HsReal t -> t -> HsReal t -> t -> t -> IO t
mkInplaceFunction op a m b x y = op x a m b x y >> pure x
mkNewFunction     op a m b x y = withEmpty x $ \r -> op r a m b x y

addmv, addmv_ :: TensorMathBlas t => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmv  = mkNewFunction     _addmv
addmv_ = mkInplaceFunction _addmv

addmm, addmm_ :: TensorMathBlas t => HsReal t -> t -> HsReal t -> t -> t -> IO t
addmm  = mkNewFunction     _addmm
addmm_ = mkInplaceFunction _addmm

addr, addr_ :: TensorMathBlas t => HsReal t -> t -> HsReal t -> t -> t -> IO t
addr  = mkNewFunction     _addr
addr_ = mkInplaceFunction _addr

addbmm, addbmm_ :: TensorMathBlas t => HsReal t -> t -> HsReal t -> t -> t -> IO t
addbmm  = mkNewFunction     _addbmm
addbmm_ = mkInplaceFunction _addbmm

baddbmm, baddbmm_ :: TensorMathBlas t => HsReal t -> t -> HsReal t -> t -> t -> IO t
baddbmm  = mkNewFunction     _baddbmm
baddbmm_ = mkInplaceFunction _baddbmm

