module Torch.Class.Tensor.Math.Blas where

import Data.Void
import Torch.Class.Types

class TensorMathBlas t where
  addmv_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
  addmm_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
  addr_        :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
  addbmm_      :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
  baddbmm_     :: t -> HsReal t -> t -> HsReal t -> t -> t -> io ()
  dot          :: t -> t -> io (HsAccReal t)

type IntTensor = Void

class GPUTensorMathBlas t where
  btrifact :: t -> IntTensor -> IntTensor -> Int -> t -> io ()
  btrisolve :: t -> t -> t -> IntTensor -> io ()


